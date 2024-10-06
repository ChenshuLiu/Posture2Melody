# Posture2Melody

## Table of Contents
1. [TLDR](#TLDR)
2. [Motivation](#motivation)
    1. [A Journey into the Interplay of Sound and Emotion](#a-journey-into-the-interplay-of-sound-and-emotion)
    2. [The Genesis of the Idea: Music as an Emotional Regulator](#the-genesis-of-the-idea-music-as-an-emotional-regulator)
    3. [Future Vision: Music as a Bridge to Emotional Harmony](#future-vision-music-as-a-bridge-to-emotional-harmony)
3. [Features](#features)
4. [Repo Structure](#repo-structure)

## TLDR
**Posture2Melody** is a project that uses a **GAN-Transformer-based architecture** to generate melodies from human postures. Inspired by the idea that body postures, specifically **expansiveness of human posture**, reflects emotional states, this project seeks to create a seamless interaction between bodily movement and music. Whether it’s through dance or everyday postures, Posture2Melody transforms these movements into melodies, potentially acting as an emotional therapy tool. By synchronizing bodily movement and music, Posture2Melody seeks to develop a creative technique that could be used in **emotional therapy**.

## Motivation

### A Journey into the Interplay of Sound and Emotion
“What if music could do more than just evoke feelings? What if it could actually shape our inner emotional landscape?” This question has driven my exploration into the interconnectedness of sound, psychological well-being, and human emotions. [Carl G. Jung](https://en.wikipedia.org/wiki/Carl_Jung), the renowned Swiss psychiatrist, held a deep belief in the power of [mandalas](https://www.britannica.com/topic/mandala-diagram) for psychological therapy. In Jung’s view, mandalas were not just artistic creations but symbolic representations of the human psyche, reflecting the inner self and serving as tools for achieving psychological balance. He often used mandalas in therapy to help individuals integrate different aspects of their consciousness and reach a state of inner harmony. **Jung believed that these geometric patterns, found across various cultures and spiritual traditions, could connect a person’s conscious and unconscious minds, leading to self-discovery and healing.**

![Carl Jung and Mandalas](https://i0.wp.com/carljungdepthpsychologysite.blog/wp-content/uploads/2014/01/65c91-temple.jpg?ssl=1)

What intrigued me even more was a discovery made by acoustic physicist John Stuart Reid ([Click this link to check out his talk](https://youtu.be/xgDg2PP_oHw)). Reid demonstrated that mandalas could be created through pure sound waves, following the principles of [_Cymatics_](https://en.wikipedia.org/wiki/Cymatics)—the study of sound wave phenomena—proposed by [Hans Jenny](https://en.wikipedia.org/wiki/Hans_Jenny_(cymatics)). Through Reid’s experiment, it became evident that the seemingly abstract mandalas of Jungian psychology could be reproduced using sound vibrations, suggesting a fascinating link between visual symbols of the mind and the physical world of acoustic waves.

![Cymatics](https://d29rinwu2hi5i3.cloudfront.net/article_media/41418aeb-8545-4283-83bc-ae806751bdfa/figure_3_-_piano_notes.jpg)

### The Genesis of the Idea: Music as an Emotional Regulator
This intersection between acoustic science and psychology reminded me of the words of Nikola Tesla: _“If you want to find the secrets of the universe, think in terms of energy, frequency, and vibration.”_ I began to wonder whether these waves hold the energy to interact with people’s emotional states. Gelder et al. [$^1$](https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/wcs.1335?casa_token=XGr12RydjjwAAAAA%3ArAzE58HAAv6WUwilhK8CPfsjF25X5QyxZeZ65S-GU5X-19A4gGMDIFXh8GjJ9AC73Pmk26qmzYDsFZY)From this foundation, I developed the idea of using music as a medium for emotional therapy, creating a project that transforms human postures into melodies, which I titled "Posture2Melody", enabling people to express their emotional states through sound. 

This concept is based on research demonstrating the connection between physical posture and emotional state, where expansive postures (e.g., standing tall with arms outstretched) are linked to confidence and positivity, while closed or less expansive postures are associated with anxiety, conservatism, or even depression. Posture2Melody translates physical postures captured by Mediapipe into musical notes using a Transformer-Adversarial Model, which is inspired by Professor [Cheng-Zhi Huang](https://czhuang.github.io/)'s paper "[Music Transformer](https://arxiv.org/abs/1809.04281)", creating a melody that mirrors an individual’s emotional state. Through this project, I aim to give people a new way to “hear” their feelings, providing an unconventional yet intuitive form of emotional expression. The hope is that when people can visualize—or in this case, _hear_—their emotions, they may be better able to understand and regulate their feelings, much like how the visualization of mandalas in Jungian therapy fosters self-awareness and personal growth.

### Future Vision: Music as a Bridge to Emotional Harmony
The potential applications of this project are immense. This system could be integrated into therapy sessions, where therapists use the generated music to better understand and engage with their patients’ emotional states. This musical journey could accompany a patient’s psychological journey, guiding them from states of imbalance to ones of equilibrium. Moreover, my vision for this project extends beyond merely using posture data. I hope to incorporate other modalities, such as vocal pitch, heart rate, or even brainwave frequencies, to generate music that resonates deeply with the person’s emotional and physical state. By drawing on a wide range of inputs, this system could create a comprehensive auditory representation of one’s mental state, enhancing its potential to aid in emotional regulation and healing.

In a world where mental health issues are becoming increasingly prevalent, we must explore new and innovative approaches to therapy. Music, with its universal appeal and intrinsic connection to human emotions, could offer a promising solution. By integrating sound waves, psychological theory, and technology, I am dedicated to uncovering the hidden potential of music as a bridge to emotional harmony. Through this exploration, I hope to unlock the healing power of sound and transform music into a tool that can lead to genuine emotional well-being.

## Features

- **GAN-Transformer Architecture**: Utilizes Generative Adversarial Networks (GANs) and Transformers to map posture data to musical melodies.
- **Emotion-Driven Design**: Explores the connection between physical posture and emotional states, creating a dynamic link between movement and music.
- **Therapeutic Potential**: The generated melodies could be used to help regulate emotional states and reduce anxiety, serving as an intervention tool for emotional therapy.

## Repo Structure
```
Posture2Melody/
│
├── mediadata/              # Sample posture data
│ ├── landmark/             # The human posture landmark data extracted from video file using `mediapipe`
│ ├── mel_spectrogram/      # The mel-spectrogram data extracted from audio file using `librosa`
├── posture2melody.ipynb    # GAN-Transformer model architecture
└── README.md               # Project overview
```