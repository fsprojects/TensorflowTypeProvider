# TensorflowTypeProvider

This Type Provider aims to eliminate the need for ‘magic strings’ associated with accessing pre-trained Tensorflow Graphs. 

This Type Provider demonstrates that the programming model can be separated from the execution and that design time shape information can be represented within the type system. 

Concretely, for now, this Type Provider should provide a more interactive and intuitive mechanism for exploring the Tensorflow Graphs than either the programmatic interrogation with Python or the nested graphical exploration with Tensorboard. 

Typed access to NPY/NPZ files is also included. 

![NPY](images/NPY.gif)

![NPZ](images/NPZ.gif)

![Tensorflow Graph](images/TF.gif)


# High level TODO
 * Improve memory usage
 * Support Proto text file
 * Support Tensorflow Checkpoints
 * Code-gen Op specific types

