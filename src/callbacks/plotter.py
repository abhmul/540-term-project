from keras.callbacks import Callback
import warnings
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class Plotter(Callback):
    """
    Plots keras training metric during training
    :param monitor: The metric to monitor and plot. Set to "loss" to plot the loss. Defaults to plotting the loss.
    :param scale: The scale to plot with. Options are "linear" and "log". Defaults to linear.
    :param plot_during_train: Will plot the metric during training of model
    :param save_to_file: Saves the plot to a file
    :param block_on_end: Blocks running code at the end of training.
    """

    def __init__(self, monitor="loss", scale='linear', plot_during_train=True, save_to_file=None, block_on_end=True):
        super().__init__()
        if plt is None:
            raise ValueError("Must be able to import Matplotlib to use the Plotter.")
        self.scale = scale
        self.monitor = monitor
        self.plot_during_train = plot_during_train
        if not self.plot_during_train:
            raise NotImplementedError("Turning off plotting during training is not implemented yet")
        self.save_to_file = save_to_file
        self.block_on_end = block_on_end
        plt.ion()
        self.fig = plt.figure()
        self.title = "{} per Epoch".format(self.monitor)
        self.xlabel = "Epoch"
        self.ylabel = self.monitor
        self.ax = self.fig.add_subplot(111, title=self.title,
                                       xlabel=self.xlabel, ylabel=self.ylabel)
        self.ax.set_yscale(self.scale)
        self.x = []
        self.y_train = []
        self.y_val = []

    def on_train_end(self, logs={}):
        plt.ioff()
        if self.block_on_end:
            plt.show()
        return

    def on_epoch_end(self, epoch, logs={}):
        # Set up the plot
        self.ax.clear()
        self.fig.suptitle(self.title)
        self.ax.set_yscale(self.scale)

        train_score = logs.get(self.monitor)
        val_score = logs.get("val_" + self.monitor)
        # Some error checking
        if train_score is None:
            warnings.warn("Monitoring metric {monitor} cannot be found, cannot plot".format(monitor=self.monitor))
            return

        # Do the train plotting
        self.x.append(len(self.x))
        self.y_train.append(train_score)
        self.ax.plot(self.x, self.y_train, 'b-')

        # Check if we have validation scores to plot
        if val_score is None:
            warnings.warn("No validation metric {monitor} can be found, cannot plot validation scores".format(monitor=self.monitor))

        else:
            # Do the val plotting
            self.y_val.append(val_score)
            self.ax.plot(self.x, self.y_val, 'g-')

        # Save file and draw
        if self.save_to_file is not None:
            self.fig.savefig(self.save_to_file)
        self.fig.canvas.draw()
        return
