import logging
import os

import numpy as np
import tqdm
from ksptrack import params
from skimage import io


def main(cfg, out_path):

    print('Writing result frames to: ' + out_path)

    res = np.load(os.path.join(out_path, 'results.npz'))

    frame_paths = {
        'bin': os.path.join(out_path, 'bin_frames'),
        'prob': os.path.join(out_path, 'prob_frames')
    }
    for v in frame_paths.values():
        if (not os.path.exists(v)):
            print('Creating output frame dir: {}'.format(v))
            os.makedirs(v)

    scores = (res['ksp_scores_mat'].astype('uint8')) * 255

    pbar = tqdm.tqdm(total=scores.shape[0])
    for i in range(scores.shape[0]):
        io.imsave(os.path.join(frame_paths['bin'], 'im_{:04d}.png'.format(i)),
                  scores[i])
        pbar.set_description('[bin frames]')
        pbar.update(1)

    pbar.close()

    if ('pm_scores_mat' in res.keys()):
        scores_pm = (res['pm_scores_mat'] * 255.).astype('uint8')
        pbar = tqdm.tqdm(total=scores.shape[0])
        for i in range(scores.shape[0]):
            io.imsave(
                os.path.join(frame_paths['prob'], 'im_{:04d}.png'.format(i)),
                scores_pm[i])
            pbar.set_description('[prob frames]')
            pbar.update(1)

        pbar.close()


if __name__ == "__main__":
    p = params.get_params()

    p.add('--out-path', required=True)

    cfg = p.parse_args()
    main(cfg, cfg.out_path)
