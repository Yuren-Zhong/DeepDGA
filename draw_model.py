
from models import basic_cnn_model, lstm_model, cnn_lstm_model, bidirectional_lstm_model
from keras.utils.vis_utils import plot_model

model = basic_cnn_model()
plot_model(model, to_file='basic_cnn_model_plot.png', show_shapes=False, show_layer_names=False)

model = lstm_model()
plot_model(model, to_file='lstm_model_plot.png', show_shapes=False, show_layer_names=False)

model = cnn_lstm_model()
plot_model(model, to_file='cnn_lstm_model_plot.png', show_shapes=False, show_layer_names=False)

model = bidirectional_lstm_model()
plot_model(model, to_file='bidirectional_lstm_model_plot.png', show_shapes=False, show_layer_names=False)