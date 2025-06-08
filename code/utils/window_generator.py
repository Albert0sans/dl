import matplotlib.pyplot as plt
import numpy as np


class WindowGenerator:
    def __init__(
        self,
        input_width,
        label_width,
        shift,
        train_df=None,
        val_df=None,
        test_df=None,
        label_columns=None,
        batch_size=32
    ):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.batch_size=batch_size
       
        self.target_indices=[self.column_indices[key] for key in self.label_columns]
        
        self.total_window_size = input_width + shift

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}",
            ]
        )

    def plot(self, models=None, plot_col=None, max_subplots_per_model=1):
        if not isinstance(models, list):
            models = [models]
       
        # For plotting, we'll still grab a single example batch
        inputs, labels = self.example
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots_per_model, len(inputs))
        num_models = len(models)
        fig, axes = plt.subplots(
            nrows=num_models,
            ncols=max_n,
            squeeze=False
        )

        for row_idx, model in enumerate(models):
            predictions = model(inputs) if model is not None else None
            for col_idx in range(max_n):
                ax = axes[row_idx][col_idx]

                # Plot inputs
                ax.plot(
                    self.input_indices,
                    inputs[col_idx, :, plot_col_index],
                    label="Inputs",
                    marker=".",
                    linestyle="--",
                    alpha=0.6,
                    zorder=-10,
                )

                # Determine label column index
                if self.label_columns:
                    label_col_index = self.label_columns_indices.get(plot_col, None)
                else:
                    label_col_index = plot_col_index

                if label_col_index is not None:
                    # Plot labels
                    label_values = labels[col_idx, :, label_col_index]
                    label_indices = self.label_indices[:label_values.shape[0]]

                    ax.scatter(
                        label_indices,
                        label_values,
                        edgecolors="k",
                        label="Labels",
                        c="#2ca02c",
                        s=64,
                        alpha=0.6,
                    )

                    # Plot predictions
                    if predictions is not None:
                        ax.scatter(
                            self.label_indices,
                            predictions[col_idx, :, label_col_index],
                            marker="X",
                            edgecolors="k",
                            label="Predictions",
                            c="#ff7f0e",
                            s=64,
                            alpha=0.6,
                        )

                if col_idx == 0:
                    ax.set_ylabel(f"Model {row_idx + 1}\n{plot_col} [normed]")
                if row_idx == num_models - 1:
                    ax.set_xlabel("Time [h]")
                if row_idx == 0 and col_idx == 0:
                    ax.legend()
                ax.grid(True)

        plt.show()

    def split_window_old(self, features):
        labels = features[self.labels_slice, :]
        if self.label_columns is not None:
            labels = np.stack(
                [labels[:, self.column_indices[name]] for name in self.label_columns],
                axis=-1,
            )
        indexnottoinclude=self.target_indices
        total_indices = features.shape[-1]

        indices_to_include = [i for i in range(total_indices) if i not in indexnottoinclude]

        inputs = features[self.input_slice, indices_to_include]

        return inputs, labels

    def split_window(self, features):
        
        
            labels = features[self.labels_slice, :]
            if self.label_columns is not None:
                labels = np.stack(
                    [labels[:, self.column_indices[name]] for name in self.label_columns],
                    axis=-1,
                )
            indexnottoinclude=self.target_indices
            total_indices = features.shape[-1]

            indices_to_include = [i for i in range(total_indices) if i not in indexnottoinclude]

            inputs = features[ self.input_slice, indices_to_include]

            return inputs, labels




    def make_dataset_generator(self, data, ):
        data = np.array(data, dtype=np.float32)
        num_windows = len(data) - self.total_window_size + 1
        
        for i in range(num_windows):
            window_data = data[i : i + self.total_window_size]
            
            
            # Yield the split window data
            yield self.split_window(window_data, )

    @property
    def train(self):
        return self.make_dataset_generator(self.train_df)

    @property
    def val(self):
        return self.make_dataset_generator(self.val_df)

    @property
    def test(self):
        return self.make_dataset_generator(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            # We take the first element from the generator for the example
            inputs, labels = next(iter(self.test))
            
            # Add a new axis to make it a batch of 1 for plotting
            inputs = tuple(inp[np.newaxis, :, :] for inp in inputs)
            labels = labels[np.newaxis, :, :]
            
            # And cache it for next time
            self._example = (inputs, labels)
            result = (inputs, labels) 
        return result