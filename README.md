# pyambisonic

A set of Python scripts written to decode Ambisonic B-Format audio files into a 5.1 srround sound format. Each stream for the 5 channels is stored in a separate .wav file.

Also included is a script (optimizeDecoderCoeffSD.py) which learns the required parameters for decoding the Ambisonics file for any symmetric 5.1 channel layout through an iterative Gradient Descent optimization of fitness functions for low and high frequencies.

References
[1] David Moore, Jonathan Wakefield "The Design and Detailed Analysis of First Order Ambisonic Decoders for the ITU Layout," Audio Engineering Society Convention Paper 7053

[2] Michael Gerzon, Geoffrey Barton, "Ambisonic Decoders for HDTV," Presented at the 92nd Audio Engineering Society Convention

[3] Daniel Arteaga "An Ambisonics decoder for irregular 3D loudspeaker arrays," Audio Engineering Society Convention Paper 8918

[4] Michael Gerzon, Geoffrey Barton "Surround Sound Apparatus," United States Patent Appl. No.: 904,440
