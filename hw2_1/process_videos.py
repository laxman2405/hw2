import numpy as np
import json
import os

def get_mappings(training_feat_dir, label_json_file, mode, min_word_count=4):
    video_filenames = os.listdir(training_feat_dir)
    video_features = {filename[:-4]: np.load(os.path.join(training_feat_dir, filename)) for filename in video_filenames}

    with open(label_json_file, 'r') as f:
        video_captions = json.load(f)

    unwanted_chars = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    captions_by_video = {
        video["id"]: [
            caption.translate(str.maketrans('', '', unwanted_chars))
            for caption in video["caption"]
        ]
        for video in video_captions
    }

    all_captions = [caption for captions in captions_by_video.values() for caption in captions]
    word_freq = {}
    for sentence in all_captions:
        for word in sentence.lower().split():
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1

    vocab = [word for word, count in word_freq.items() if count >= min_word_count]
    print(f'Filtered vocabulary: {len(vocab)} words (min count: {min_word_count}) from {len(word_freq)} total words.\n')

    special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
    word_to_id = {token: idx for idx, token in enumerate(special_tokens)}
    id_to_word = {idx: token for token, idx in word_to_id.items()}

    for idx, word in enumerate(vocab, start=len(special_tokens)):
        word_to_id[word] = idx
        id_to_word[idx] = word

    total_captions = len(all_captions)
    unique_captions = len(set(all_captions))
    captions_per_video = {video_id: len(captions) for video_id, captions in captions_by_video.items()}
    max_captions_per_video = max(captions_per_video.values())
    feature_shape = next(iter(video_features.values())).shape if video_features else (0, 0)

    print(f"Video feature shape: {feature_shape}")
    print(f"Total captions: {total_captions}")
    print(f"Unique captions: {unique_captions}")
    print(f"Max captions for a single video: {max_captions_per_video}")

    return word_to_id, id_to_word
