import sys, time

def main():
    import torch
    torch.set_num_threads(4)
    from totalsegmentator.python_api import totalsegmentator
    for c in sys.argv[1:]:
        t = time.time()
        totalsegmentator(f'nii/{c}_ct2.nii.gz', f'nii/{c}_seg.nii.gz', ml=True, fast=True, device='cpu',
                         nr_thr_resamp=1, nr_thr_saving=1, quiet=False)
        print(c, 'done', round(time.time() - t), 's', flush=True)

if __name__ == '__main__':
    main()
