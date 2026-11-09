#!/usr/bin/env python3
"""Use the preference plotter with every supported CNN model name."""
import plot_preference_calibrated as plot

plot.NAMES.update({'resnet34':'ResNet34','resnet50':'ResNet50',
                   'vgg19':'VGG19','alexnet':'AlexNet'})

if __name__=='__main__':plot.main()
