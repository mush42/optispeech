# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

hparams = dict(
    # Model
    arch=['sep5', 'sep25', 'sep13', 'sep9', 'sep17', 'sep21', 'sep9', 'sep13',],
    hidden_size=256,
    filter_size=1024,
    enc_layers=4,
    dec_layers=4,
    ffn_kernel_size=9,
    activation='activation',
    dropout=0.2,
    # predictor
    predictor_hidden=256,
    predictor_sg=True,
    predictor_layer_type='sepconv',
    # duration
    dur_predictor_kernel=3,
    use_gt_dur=False,
    sep_dur_loss=True,
    dur_predictor_layer=2,
    # pitch and energy
    use_pitch_embed=True,
    use_uv=True,
    use_energy_embed=False,
    pitch_loss='l1',
    pitch_predictor_layer=5,
    # loss
    lambda_dur=1.0,
    lambda_pitch=1.0,
    lambda_uv=1.0,
    lambda_energy=1.0,
    mel_loss='l1'
)
