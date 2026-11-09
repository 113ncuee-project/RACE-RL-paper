#!/usr/bin/env python3
"""Summarize independent and shared preferences across all supported CNNs."""
import summarize_preference_direction as summary

summary.NAMES.update({'resnet34':'ResNet34','resnet50':'ResNet50',
                      'vgg19':'VGG19','alexnet':'AlexNet'})

if __name__=='__main__':summary.main()
