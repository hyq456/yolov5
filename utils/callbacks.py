# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Callback utils
"""


class Callbacks:
    """"
    Handles all registered callbacks for YOLOv5 Hooks
    """

<<<<<<< HEAD
=======
    # Define the available callbacks
>>>>>>> f01eeeed0c60ee4d6765925190c3e910d115a187
    _callbacks = {
        'on_pretrain_routine_start': [],
        'on_pretrain_routine_end': [],

        'on_train_start': [],
        'on_train_epoch_start': [],
        'on_train_batch_start': [],
        'optimizer_step': [],
        'on_before_zero_grad': [],
        'on_train_batch_end': [],
        'on_train_epoch_end': [],

        'on_val_start': [],
        'on_val_batch_start': [],
        'on_val_image_end': [],
        'on_val_batch_end': [],
        'on_val_end': [],

        'on_fit_epoch_end': [],  # fit = train + val
        'on_model_save': [],
        'on_train_end': [],

        'teardown': [],
    }

<<<<<<< HEAD
    def __init__(self):
        return

=======
>>>>>>> f01eeeed0c60ee4d6765925190c3e910d115a187
    def register_action(self, hook, name='', callback=None):
        """
        Register a new action to a callback hook

        Args:
            hook        The callback hook name to register the action to
<<<<<<< HEAD
            name        The name of the action
=======
            name        The name of the action for later reference
>>>>>>> f01eeeed0c60ee4d6765925190c3e910d115a187
            callback    The callback to fire
        """
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        assert callable(callback), f"callback '{callback}' is not callable"
        self._callbacks[hook].append({'name': name, 'callback': callback})

    def get_registered_actions(self, hook=None):
        """"
        Returns all the registered actions by callback hook

        Args:
            hook The name of the hook to check, defaults to all
        """
        if hook:
            return self._callbacks[hook]
        else:
            return self._callbacks

<<<<<<< HEAD
    def run_callbacks(self, hook, *args, **kwargs):
        """
        Loop through the registered actions and fire all callbacks
        """
        for logger in self._callbacks[hook]:
            # print(f"Running callbacks.{logger['callback'].__name__}()")
            logger['callback'](*args, **kwargs)

    def on_pretrain_routine_start(self, *args, **kwargs):
        """
        Fires all registered callbacks at the start of each pretraining routine
        """
        self.run_callbacks('on_pretrain_routine_start', *args, **kwargs)

    def on_pretrain_routine_end(self, *args, **kwargs):
        """
        Fires all registered callbacks at the end of each pretraining routine
        """
        self.run_callbacks('on_pretrain_routine_end', *args, **kwargs)

    def on_train_start(self, *args, **kwargs):
        """
        Fires all registered callbacks at the start of each training
        """
        self.run_callbacks('on_train_start', *args, **kwargs)

    def on_train_epoch_start(self, *args, **kwargs):
        """
        Fires all registered callbacks at the start of each training epoch
        """
        self.run_callbacks('on_train_epoch_start', *args, **kwargs)

    def on_train_batch_start(self, *args, **kwargs):
        """
        Fires all registered callbacks at the start of each training batch
        """
        self.run_callbacks('on_train_batch_start', *args, **kwargs)

    def optimizer_step(self, *args, **kwargs):
        """
        Fires all registered callbacks on each optimizer step
        """
        self.run_callbacks('optimizer_step', *args, **kwargs)

    def on_before_zero_grad(self, *args, **kwargs):
        """
        Fires all registered callbacks before zero grad
        """
        self.run_callbacks('on_before_zero_grad', *args, **kwargs)

    def on_train_batch_end(self, *args, **kwargs):
        """
        Fires all registered callbacks at the end of each training batch
        """
        self.run_callbacks('on_train_batch_end', *args, **kwargs)

    def on_train_epoch_end(self, *args, **kwargs):
        """
        Fires all registered callbacks at the end of each training epoch
        """
        self.run_callbacks('on_train_epoch_end', *args, **kwargs)

    def on_val_start(self, *args, **kwargs):
        """
        Fires all registered callbacks at the start of the validation
        """
        self.run_callbacks('on_val_start', *args, **kwargs)

    def on_val_batch_start(self, *args, **kwargs):
        """
        Fires all registered callbacks at the start of each validation batch
        """
        self.run_callbacks('on_val_batch_start', *args, **kwargs)

    def on_val_image_end(self, *args, **kwargs):
        """
        Fires all registered callbacks at the end of each val image
        """
        self.run_callbacks('on_val_image_end', *args, **kwargs)

    def on_val_batch_end(self, *args, **kwargs):
        """
        Fires all registered callbacks at the end of each validation batch
        """
        self.run_callbacks('on_val_batch_end', *args, **kwargs)

    def on_val_end(self, *args, **kwargs):
        """
        Fires all registered callbacks at the end of the validation
        """
        self.run_callbacks('on_val_end', *args, **kwargs)

    def on_fit_epoch_end(self, *args, **kwargs):
        """
        Fires all registered callbacks at the end of each fit (train+val) epoch
        """
        self.run_callbacks('on_fit_epoch_end', *args, **kwargs)

    def on_model_save(self, *args, **kwargs):
        """
        Fires all registered callbacks after each model save
        """
        self.run_callbacks('on_model_save', *args, **kwargs)

    def on_train_end(self, *args, **kwargs):
        """
        Fires all registered callbacks at the end of training
        """
        self.run_callbacks('on_train_end', *args, **kwargs)

    def teardown(self, *args, **kwargs):
        """
        Fires all registered callbacks before teardown
        """
        self.run_callbacks('teardown', *args, **kwargs)
=======
    def run(self, hook, *args, **kwargs):
        """
        Loop through the registered actions and fire all callbacks

        Args:
            hook The name of the hook to check, defaults to all
            args Arguments to receive from YOLOv5
            kwargs Keyword Arguments to receive from YOLOv5
        """

        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"

        for logger in self._callbacks[hook]:
            logger['callback'](*args, **kwargs)
>>>>>>> f01eeeed0c60ee4d6765925190c3e910d115a187
