# IWSLT-2020: Video translation task

Task website: http://iwslt.org/doku.php?id=video_speech_translation

## How to get data?

+ Requirement: git and git LFS [[Installation](https://help.github.com/en/github/managing-large-files/installing-git-large-file-storage)]
+ Command: ```
git clone https://github.com/nguyenbh/iwslt2020_video_translation.git```

## Descriptions

### Chinese-English dev set
This set contains 5 video clips of Chinese speakers. Total duration is about 1 hour, and all clips are in e-Commerce domain.

The folder structure is organized as follow
```
clip_id\
    clip_id.short.dev.mp4: raw video file
    clip_id.short.dev.wav: raw wav file
    clip_id.short.dev.transcription_2ref.txt: manual Chinese transcription and 2 human English translations.
    clip_id.cvte.out: output of the baseline kaldi CVTE Madanrin model 
    wav\: wav files contain manual transcription.
```
