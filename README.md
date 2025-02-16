# Steering Language Model to Stable Speech Emotion Recognition via Contextual Perception and Chain of Thought 
<sub>The official implementation of C$^2$SER (ACL 2025) </sub>

## Abstract
we propose $\mathbf{C^2SER}$, a novel ALM designed to enhance the stability and accuracy of SER through \textbf{C}ontextual perception and \textbf{C}hain of Thought (CoT). C$^2$SER integrates the Whisper encoder for semantic perception and Emotion2Vec-S for acoustic perception, where Emotion2Vec-S extends Emotion2Vec with semi-supervised learning to enhance emotional discrimination. Additionally, C$^2$SER employs a CoT approach, processing SER in a step-by-step manner while leveraging speech content and speaking styles to improve recognition. To further enhance stability, C$^2$SER introduces self-distillation from explicit CoT to implicit CoT, mitigating error accumulation and boosting recognition accuracy. Extensive experiments show that C$^2$SER outperforms existing popular ALMs, such as Qwen2-Audio and SECap, delivering more stable and precise emotion recognition. We release the training code, checkpoints, and test sets to facilitate further research

