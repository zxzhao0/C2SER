# Steering Language Model to Stable Speech Emotion Recognition via Contextual Perception and Chain of Thought <br> <sub> The official implementation of C<sup>2</sup>SER (submit to ACL 2025) </sub>

## Abstract
We propose C<sup>2</sup>SER, a novel ALM designed to enhance the stability and accuracy of SER through **C**ontextual perception and **C**hain of Thought (CoT). C<sup>2</sup>SER integrates the Whisper encoder for semantic perception and Emotion2Vec-S for acoustic perception, where Emotion2Vec-S extends Emotion2Vec with semi-supervised learning to enhance emotional discrimination. Additionally, C<sup>2</sup>SER employs a CoT approach, processing SER in a step-by-step manner while leveraging speech content and speaking styles to improve recognition. To further enhance stability, C<sup>2</sup>SER introduces self-distillation from explicit CoT to implicit CoT, mitigating error accumulation and boosting recognition accuracy. Extensive experiments show that C<sup>2</sup>SER outperforms existing popular ALMs, such as Qwen2-Audio and SECap, delivering more stable and precise emotion recognition.

## Roadmap

C<sup>2</sup>SER is designed to mitigate hallucinations in speech emotion recognition (SER) and to deliver stable emotion recognition. C<sup>2</sup>SER architecture consists of two primary components: a contextual perception module and a text-based large language model (LLM). The contextual perception module extracts detailed information regarding both the semantic and acoustic aspects, which the text LLM subsequently leverages via a chain-of-thought process to make final predictions.

More specifically, the contextual perception module comprises the following elements: a Whispe encoder for semantic perception, Emotion2Vec-S for acoustic perception, and a connection model designed to align the feature dimensions with those required by the text LLM.

<p align="center">
  <img src="figs/details of CSER.drawio.jpg" width="500"/>
</p>

## TODO 📝
- [x] Release EMO-Emilia test set
- [ ] Release pretrained model
- [ ] Release Inference pipeline
- [ ] More to be added

### The EMO-Emilia Dataset

To overcome the limitations of existing SER datasets—inconsistent quality, low diversity, and lack of real-world authenticity—we introduce a new SER test set, **Emo-Emilia**. 
Specifically, we apply the automated labeling approach to annotate Emilia, a large-scale multilingual and diverse speech generation resource with over 100,000 hours of speech data that captures a wide range of emotional contexts.
We then manually verify the accuracy of the emotion labels. Each utterance is checked by at least two experts to ensure both accuracy and reliability. The final proposed test set, Emo-Emilia, consists of 1400 test samples, with 100 samples per emotion category across seven types in both Chinese and English (700 samples per language).

The original Emilia dataset can be accessed here [Emilia](https://emilia-dataset.github.io/Emilia-Demo-Page/).

EMO-Emilia Dataset files: './EMO-Emilia/EMO-Emilia-ALL.jsonl'



### C<sup>2</sup>SER

The training pipline will coming soon!

### Emotion2Vec-S

The pre-trained model will coming soon!