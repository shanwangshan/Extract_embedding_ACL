import numpy as np
import scipy
import yaml
import librosa
import torch
import models.audio_models as audio_mod
import models.resnetAudio as resnet_mod
import models.MLP_head as mlp
import argparse

parser = argparse.ArgumentParser(description='Code for extracting embeddings')

parser.add_argument('-p', '--params_yaml', dest='params_yaml', action='store', required=True, type=str)

config = parser.parse_args()
args = yaml.full_load(open(config.params_yaml))

def get_normalized_audio(y, head_room=0.005):

    mean_value = np.mean(y)
    y -= mean_value

    max_value = max(abs(y)) + head_room
    return y / max_value

def get_mel_spectrogram_lib(audio):

    audio = audio.reshape([1, -1])

    window = scipy.signal.hamming(args['win_length_samples'], sym=False)

    mel_spectrogram = librosa.feature.melspectrogram(y=audio[0, :],
                                                     sr=args['fs'],
                                                     win_length=args['win_length_samples'],
                                                     hop_length= args['hop_length_samples'],
                                                     window=window,
                                                     n_fft=args['n_fft'],
                                                     n_mels=args['n_mels'],
                                                     center=True,
                                                     # center=False,
                                                     power=2
                                                     ).T


    mel_spectrogram = np.log10(mel_spectrogram + 1.1e-08)

    return mel_spectrogram


y, sr = librosa.load(args['audio_path'])
print('loaded audio file', args['audio_path'])
data = get_normalized_audio(y)
data = np.reshape(data, [-1, 1])
mel_spec = get_mel_spectrogram_lib(data)
mel_spec=mel_spec[:(mel_spec.shape[0]//100)*100,:]
mel_spec = torch.from_numpy(mel_spec).view(-1,100,args['n_mels'])
mel_spec = mel_spec[:,None,:,:]


use_cuda = torch.cuda.is_available()
print('use_cude',use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")

model = resnet_mod.resnet18(args, num_classes=20).to(device)
model_dir = args['model_path']
#model.load_state_dict(torch.load(model_dir))

# use this line if cuda is not available
model.load_state_dict(torch.load(model_dir,map_location=torch.device('cpu')))
print('loaded model from ', model_dir)


model.eval()
mel_spec= mel_spec.to(device)

with torch.no_grad():
    embeddings,_ = model(mel_spec)


print('extracting done and the embedding shape is', embeddings.size())
