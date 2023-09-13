## To-do

* implement test split for classifier

## To-do

* deterministic sampling
  * override `get_train_sampler` function
  * (done) not necessary if use `predict` function
* save model output (callback, metrics, loss function)
  * callback (x) does not have output
  * metrics (x) only run during evalute loop <https://stackoverflow.com/questions/68806265/huggingface-trainer-logging-train-data>
  * loss function (x) dealing with randomness / predict calls this
  * (done) use `predict` function
* generate masks (change `transform`` function) also deterministic for training
  * (done) by adding transform function.
* custom loss function (define new `Trainer` class)
  * (done) defined new model class
* model output
  * use hidden states
* evaluate multiple times
  * first evaluate
