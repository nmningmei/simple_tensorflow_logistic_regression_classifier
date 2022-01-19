# This is a simple logistic regression classifier implemented in tensorflow 2.0

```
tf.random.set_seed(12345)
    logistic_regression = build_logistic_regression(
                        input_size              = X_train.shape[1],
                        output_size             = 2,
                        special                 = False,
                        kernel_regularizer      = regularizers.L2(l2 = 1e-3),
                        activity_regularizer    = regularizers.L1(l1 = 1e-3),
                        print_model             = True,
                        )
# compile the model
logistic_regression,callbacks = compile_logistic_regression(
                                logistic_regression,
                                model_name      = 'temp.h5',
                                optimizer       = None,
                                loss_function   = None,
                                metric          = None,
                                callbacks       = None,
                                learning_rate   = 1e-3,
                                tol             = 1e-4,
                                patience        = 10,
                                )
```
