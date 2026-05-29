"""
RSMI training: coarse-grainer, critics, MI bounds, and training loop.
"""

import json
import os
import datetime

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import regularizers
from tqdm.auto import tqdm

tfd = tfp.distributions
tfkl = tf.keras.layers


class TimeHistory(tf.keras.callbacks.Callback):
    """Callback that prints logs at end of each epoch."""

    def on_epoch_end(self, epoch, logs=None):
        print(logs)



def mlp(
    hidden_dim,
    output_dim,
    layers,
    activation,
):
    hidden = [
        [
            tfkl.Dense(
                hidden_dim,
                activation,
                bias_initializer=tf.keras.initializers.HeNormal(),
            )
        ]
        for _ in range(layers)
    ]
    hidden = [item for sublist in hidden for item in sublist]
    return tf.keras.Sequential(
        hidden + [tfkl.Dense(output_dim)]
    )


def infonce_lower_bound(x, y, f_ansatz, training=False):
    """InfoNCE lower bound for I(X:Y)."""
    scores = f_ansatz(x, y, training=training)
    n = tf.shape(scores)[0]
    positive_mask = tf.eye(n, dtype=tf.bool)
    return tfp.vi.mutual_information.lower_bound_info_nce(
        logu=scores, joint_sample_mask=positive_mask
    )


lowerbounds = {
    "infonce": infonce_lower_bound,
}


class SeparableCritic(tf.keras.Model):
    """Separable critic for MI bound: f(V,E) = V_net(V).T @ E_net(E)."""

    def __init__(
        self,
        hidden_dim_V,
        hidden_dim_E,
        embed_dim,
        layers_V,
        layers_E,
        activation,
        **extra_kwargs,
    ):
        super(SeparableCritic, self).__init__()
        self.critic_name = "SeparableCritic"
        self._V = mlp(
            hidden_dim_V,
            embed_dim,
            layers_V,
            activation
        )
        self._E = mlp(
            hidden_dim_E,
            embed_dim,
            layers_E,
            activation
        )

    def call(self, V, E, training):
        return tf.einsum(
            "ij,kj->ik",
            self._V(V, training=training),
            self._E(E, training=training),
        )


class CoarseGrainer(tf.keras.Model):
    """Coarse-grainer: maps V to a soft binary H via a fully-connected encoder, a
    BatchNorm with hard MaxNorm constraint on gamma (the information bottleneck),
    and a Relaxed-Bernoulli sampling head with annealed temperature. An optional
    symmetrization wrapper around the encoder enforces a chosen one-dimensional
    irrep of the lattice/internal symmetry group."""

    def __init__(
        self,
        layers,
        batchNorm_scale,
        activation="relu",
        layer_width=None,
        num_hiddens = 1,
        kernel_regularization_weight=0,
        bias_regularization_weight=0,
        activity_regularization_weight=0,
        relaxation_rate=0.01,
        min_temperature=0.05,
        init_temperature=2,
        cg_name="",
        cur_critic="",
        dataset_path="",
        learn_beta=True,
        symmetrization_operation=lambda model: model,
    ):
        
        assert ((layers > 0) == (layer_width is not None))

        super(CoarseGrainer, self).__init__()
        self.cg_name = cg_name

        self.r = tf.constant(relaxation_rate, dtype=tf.float32)
        self.min_tau = tf.constant(min_temperature, dtype=tf.float32)
        self.init_tau = tf.constant(init_temperature, dtype=tf.float32)
        self._global_step = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        self.encoder = tf.keras.models.Sequential()
        self.gumbel_head = tf.keras.models.Sequential()
        self.dataset_path = dataset_path
        self.cg_name = cg_name
        self.cur_critic = cur_critic
        if layers == 0: # linear CG
            self.encoder.add(
                tf.keras.layers.Dense(
                    num_hiddens,
                    activation=None,
                    use_bias=True,
                    kernel_regularizer=regularizers.l2(l2=kernel_regularization_weight),
                )
            )
        else:
            for i in range(layers):
                self.encoder.add(
                    tf.keras.layers.Dense(
                        layer_width,
                        activation=activation,
                        kernel_regularizer=regularizers.l2(
                            l2=kernel_regularization_weight
                        ),
                        bias_regularizer=regularizers.l2(
                            l2=bias_regularization_weight
                        ),
                        kernel_initializer=tf.keras.initializers.HeNormal(),
                    )
                )
            self.encoder.add(
                tf.keras.layers.Dense(
                    num_hiddens,
                    kernel_regularizer=regularizers.l2(
                        l2=kernel_regularization_weight
                    ),
                    bias_regularizer=regularizers.l2(
                        l2=bias_regularization_weight
                    ),
                    kernel_initializer=tf.keras.initializers.HeNormal(),
                )
            )
    
            self.gumbel_head.add(
                tf.keras.layers.BatchNormalization(
                    axis=1,
                    scale=True,
                    center=learn_beta,
                    gamma_constraint=tf.keras.constraints.MaxNorm(
                        max_value=batchNorm_scale
                    ),
                )
            )

        self.gumbel_head.add(
            tf.keras.layers.ActivityRegularization(l2=activity_regularization_weight)
        )
        self.gumbel_head.add(
            tfkl.Lambda(
                lambda x: tf.math.multiply(
                    tf.math.subtract(
                        tfd.RelaxedBernoulli(self.tau, logits=x).sample(), 0.5
                    ),
                    2,
                )
            )
        )
        self.encoder = symmetrization_operation(self.encoder)

    def call(self, V, training):
        return self.gumbel_head(self.encoder(V, training=training),training=training)

    @property
    def global_step(self):
        # Return Python float for logging convenience
        return float(self._global_step.numpy())

    @global_step.setter
    def global_step(self, step):
        # Assign into TF variable so @tf.function sees updates
        self._global_step.assign(tf.cast(step, tf.float32))

    @property
    def tau(self):
        # TF-native annealing schedule: tensor that updates when _global_step updates
        return tf.maximum(self.min_tau, self.init_tau * tf.exp(-self.r * self._global_step))
    


def train_RSMI_optimiser(
    cg,
    CG_params,
    critic_params,
    opt_params,
    additional_data,
    train_dataset,
    log_dir_root,
    cur_critic,
    symmetrization_operation = lambda model: model,
    bound="infonce",
    test_dataset=None,
    dataset_dir = ""
):
    """Main training loop for maximisation of RSMI [I(H:E)]."""

    # Disable TF32 only inside the training entry point so that simply importing
    # RSMI does not silently reconfigure the user's TF runtime. TF32 can cause
    # numerical instability on Ampere+ GPUs for the InfoNCE bound.
    tf.config.experimental.enable_tensor_float_32_execution(False)

    assert test_dataset is not None, "train_RSMI_optimiser requires a test_dataset"
    train_dataset = train_dataset.repeat(opt_params["iterations"])
    test_dataset = test_dataset.repeat()

    log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir_root, log_dir)
    os.makedirs(log_path, exist_ok=True)

    def _json_default(obj):
        if hasattr(obj, "numpy"):
            return obj.numpy().tolist()
        if hasattr(obj, "shape") and hasattr(obj, "tolist") and len(obj.shape) > 0:
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    config = {
        "CG_params": CG_params,
        "opt_params": opt_params,
        "critic_params": critic_params,
        "bound": bound,
        "dataset_dir": dataset_dir,
        "log_dir": log_dir
    }
    
    config["V_indices"] = [list(p) for p in additional_data["V_indices"]]

    with open(os.path.join(log_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=_json_default)

    train_writer = tf.summary.create_file_writer(
        os.path.join(log_path, "train")
    )
    test_writer = tf.summary.create_file_writer(
        os.path.join(log_path, "test")
    )
    test_dataset_iter = iter(test_dataset)

    num_of_iterations = int(
        opt_params["iterations"]
        * np.ceil(opt_params["N_samples"] / opt_params["batch_size"])
    )
    cycles_per_iteration = int(
        np.ceil(opt_params["N_samples"] / opt_params["batch_size"])
    )


    CG = cg(**CG_params, symmetrization_operation=symmetrization_operation)

    f_ansatz = cur_critic(**critic_params)

   
    if (
        CG_params.get("kernel_regularization_weight", 0) != 0
        or CG_params.get("bias_regularization_weight", 0) != 0
        or CG_params.get("activity_regularization_weight", 0) != 0
    ):
        do_regularization = True
    else:
        do_regularization = False

    opt = tf.keras.optimizers.Adam(opt_params["learning_rate"])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="test_mi", patience=opt_params["patience"], verbose=1, mode="max"
    )
    callbacks = tf.keras.callbacks.CallbackList([early_stop, TimeHistory()], add_history=True, model=f_ansatz)

    do_noise = opt_params.get("do_noise", False)
    if do_noise:
        noise = opt_params["noise"]
        g_noise = tf.keras.layers.GaussianNoise(noise)

    V, E = next(test_dataset_iter)
    V = CG(V, training=True)
    mi = lowerbounds[bound](V, E, f_ansatz, training=True)

    trainable_vars = []
    if isinstance(CG, tf.keras.Model):
        trainable_vars += CG.trainable_variables
    if isinstance(f_ansatz, tf.keras.Model):
        trainable_vars += f_ansatz.trainable_variables

    @tf.function
    def train_step(V, E):
        with tf.GradientTape() as tape:
            V = CG(V, training=True)
            if do_noise:
                V = g_noise(V, training=True)
            mi = lowerbounds[bound](V, E, f_ansatz, training=True)
            loss = -mi
            if do_regularization:
                loss = loss + tf.add_n(CG.losses)
            grads = tape.gradient(loss, trainable_vars)
            opt.apply_gradients(zip(grads, trainable_vars))
        return mi, loss

    @tf.function
    def run_test(V_test, E_test, CG, f_ansatz):
        res = CG(V_test, training=False)
        mi_test = lowerbounds[bound](res, E_test, f_ansatz, training=False)
        return mi_test

    estimates = np.zeros(num_of_iterations)
    estimates_test = np.zeros(num_of_iterations)
    pbar = tqdm(total=num_of_iterations, desc="")
    i = 0
    avg_window_size = 100

    callbacks.on_train_begin()
    callbacks.on_epoch_begin(0)

    try:
        for V, E in train_dataset:
            CG.global_step = i
            if i > 0 and i % cycles_per_iteration == 0:
                estimates_test_epoch = estimates_test[
                    i - cycles_per_iteration : i
                ]
                callbacks.on_epoch_end(
                    int(i / cycles_per_iteration) - 1,
                    logs={
                        "test_mi": np.mean(estimates_test_epoch),
                        "test_mi_std": np.std(estimates_test_epoch),
                    },
                )
                if getattr(f_ansatz, "stop_training", False):
                    print("Early Stopping...")
                    break
                callbacks.on_epoch_begin(int(i / cycles_per_iteration))

            callbacks.on_batch_begin(int(i % cycles_per_iteration))
            callbacks.on_train_batch_begin(int(i % cycles_per_iteration))
            mi, loss = train_step(V, E)

            callbacks.on_train_batch_end(
                int(i % cycles_per_iteration), logs={"train_loss": -mi}
            )
            callbacks.on_batch_end(
                int(i % cycles_per_iteration), logs={"train_loss": -mi}
            )
            with train_writer.as_default():
                tf.summary.scalar("mi", data=mi, step=i)
            if np.isnan(mi.numpy()):
                print("Got NaN!")
                break

            estimates[i] = mi.numpy()
            V_test, E_test = next(test_dataset_iter)
            callbacks.on_batch_begin(int(i % cycles_per_iteration))
            callbacks.on_test_batch_begin(int(i % cycles_per_iteration))
            mi_test = run_test(V_test, E_test, CG, f_ansatz)
            callbacks.on_test_batch_end(
                int(i % cycles_per_iteration),
                logs={"test_loss": -mi_test},
            )
            callbacks.on_batch_end(
                int(i % cycles_per_iteration),
                logs={"test_loss": -mi_test},
            )
            with test_writer.as_default():
                tf.summary.scalar("mi", data=mi_test, step=i)
            estimates_test[i] = mi_test.numpy()
            if i > avg_window_size:
                pbar.set_description(
                    f"tau={CG.tau.numpy():.2f}, I={np.mean(estimates[i-avg_window_size:i]):.4f}, "
                    f"test={np.mean(estimates_test[i-avg_window_size:i]):.4f}"
                )
            pbar.update(1)
            i += 1
    except KeyboardInterrupt:
        estimates = estimates[:i]
        estimates_test = estimates_test[:i]
    if i > 0:
        start = max(0, i - cycles_per_iteration)
        estimates_test_epoch = estimates_test[start:i]
        callbacks.on_epoch_end(
            int(i / cycles_per_iteration) - 1,
            logs={
                "test_mi": np.mean(estimates_test_epoch),
                "test_mi_std": np.std(estimates_test_epoch),
            },
        )

    callbacks.on_train_end()
    print("Training complete.")

    CG.encoder.save(os.path.join(log_path, "Encoder"))

    return CG, log_dir
