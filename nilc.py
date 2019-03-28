import sys
import gc
import os
import shutil
import numpy as np
import healpy as hp
import scipy.linalg as lng
import needlets as ndl
import argparse
import time
import json

try:
    import configparser
except ImportError:
    import ConfigParser as configparser


def main(argv=None):

    if argv is None:

        argv = sys.argv

        numargs = len(sys.argv)

        if numargs < 3 :
            print( '\nEnter a Configuration File Name' )
            print( 'Syntax : python nilc.py -c config_file.ini ' )
            print( 'Quitting!!' )
            sys.exit()

        start = time.time()

        conf_parser = argparse.ArgumentParser(
                description=__doc__, # printed with -h/--help
                formatter_class=argparse.RawDescriptionHelpFormatter,
                add_help=False
                )

        conf_parser.add_argument("-c", "--conf_file",
                help="Specify config file", metavar="FILE")

        args, remaining_argv = conf_parser.parse_known_args()

        defaults = { "work_dir":"NILC_OUT/",
                     "input_dir":"input/",
                     "noise_dir":"input/Noise/",
                     "channels":"",
                     "inmaps":"",
                     "noises":"",
                     "prior_cl":"",
                     "transfer_function":"",
                     "band_centers":"",
                     "band_resolutions":"",
                     "beams":"",
                     "beam_files":"",
                     "bl_frac":"0.01",
                     "pol":"IQU",
                     "nside_out":512,
                     "lmax_out":512,
                     "smooth_beam":0,
                     "ilc_bias":1e-3,
                     "neighborhood":2,
                     "first_scale":"1",
                     "last_window":0,
                     "aic_mod":1,
                     "write_covariance":False,
                     "do_foregrounds":False,
                     "project_fg":True,
                     "weights_dir":"",
                     "masking":False,
                     "mask_file":"",
                     "mask_size":10.,
                     "mask_transition":5.,
                     "do_ludwig":False,
                     "ludwig_maps":"",
                     "ludwig_noises":"",
                     "ludwig_beams":"",
                     "cmbs":"",
                     "firbs":"",
                     "pfrac":1. }

        confile = args.conf_file

        if args.conf_file:
            config = configparser.ConfigParser()
            config.read([args.conf_file])
            defaults.update(dict(config.items("Defaults")))

        parser = argparse.ArgumentParser(
                parents=[conf_parser]
                )
        parser.set_defaults(**defaults)
        parser.add_argument("--work_dir")
        parser.add_argument("--input_dir")
        parser.add_argument("--noise_dir")
        parser.add_argument("--channels")
        parser.add_argument("--inmaps")
        parser.add_argument("--noises")
        parser.add_argument("--prior_cl")
        parser.add_argument("--transfer_function")
        parser.add_argument("--band_centers")
        parser.add_argument("--band_resolutions")
        parser.add_argument("--beams")
        parser.add_argument("--beam_files")
        parser.add_argument("--bl_frac", type=float)
        parser.add_argument("--pol")
        parser.add_argument("--nside_out", type=int)
        parser.add_argument("--lmax_out", type=int)
        parser.add_argument("--smooth_beam", type=float)
        parser.add_argument("--ilc_bias", type=float)
        parser.add_argument("--neighborhood", type=int)
        parser.add_argument("--first_scale")
        parser.add_argument("--last_window", type=int)
        parser.add_argument("--aic_mod", type=int)
        parser.add_argument("--write_covariance", type=ndl.str2bool)
        parser.add_argument("--do_foregrounds", type=ndl.str2bool)
        parser.add_argument("--project_fg", type=ndl.str2bool)
        parser.add_argument("--weights_dir")
        parser.add_argument("--masking", type=ndl.str2bool)
        parser.add_argument("--mask_file")
        parser.add_argument("--mask_size", type=float)
        parser.add_argument("--mask_transition", type=float)
        parser.add_argument("--do_ludwig", type=ndl.str2bool)
        parser.add_argument("--ludwig_maps")
        parser.add_argument("--ludwig_noises")
        parser.add_argument("--cmbs")
        parser.add_argument("--firbs")
        parser.add_argument("--pfrac", type=float)
        args = parser.parse_args(remaining_argv)

        gauss_bands=False
        bl_in=False
        k_min=0
        kfirst=1
        nfirst=0
        npol=3
        do_weight=True
        read_weight = False
        overwrite_ludwig=True
        overwrite_noise=True
        overwrite_beam=True
        add_cmb=False
        add_cib=False
        use_prior=False
        use_fl = False

        infold = str(args.input_dir).strip()
        if infold[-1] != '/' :
            infold = infold+'/'
        noisefold = str(args.noise_dir).strip()
        if noisefold[-1] != '/' :
            noisefold = noisefold+'/'
        workfold = str(args.work_dir).strip()
        if workfold[-1] != '/' :
            workfold = workfold+'/'
        dets = str(args.channels)
        dets = dets.split(',')
        dets = list(map(str.strip, dets))
        dets = np.array(dets)
        inmaps = str(args.inmaps)
        inmaps = inmaps.split(',')
        inmaps = list(map(str.strip, inmaps))
        inmaps = np.array(inmaps)
        noises = str(args.noises)
        noises = noises.split(',')
        noises = list(map(str.strip, noises))
        noises = np.array(noises)
        prior_cl = str(args.prior_cl).strip()
        transfer_function = str(args.transfer_function).strip()
        beams = str(args.beams)
        beams = beams.split(',')
        beams = list(map(str.strip, beams))
        if beams[0] != "" :
            beams = np.array(beams).astype(np.float)
        bls = str(args.beam_files)
        bls = bls.split(',')
        bls = list(map(str.strip, bls))
        bls = np.array(bls)
        if len(bls) > 1 :
            bl_in = True
        bl_min = float(args.bl_frac)
        band_res = str(args.band_resolutions)
        band_res = band_res.split(',')
        band_res = list(map(str.strip, band_res))
        if band_res[0] != "" :
            band_res = np.array(band_res).astype(np.float)
            gauss_bands = True
        cntrs = str(args.band_centers)
        cntrs = cntrs.split(',')
        cntrs = list(map(str.strip, cntrs))
        if cntrs[0] != "" :
            cntrs = np.array(cntrs).astype(np.int)
        elif not gauss_bands :
            print('\nBand Centers not Defined !!!')
            print('Define either \"band_centers\" (for cosine bands) or')
            print('              \"band_resolutions\" (for gaussian bands) in parameter file !!!')
            print( 'Quitting !!!' )
            sys.exit()
        lbeams = str(args.ludwig_beams)
        lbeams = lbeams.split(',')
        lbeams = list(map(str.strip, lbeams))
        lbeams = np.array(lbeams)
        if len(lbeams) > 1 :
            overwrite_beam = False
            lbeams = np.array(lbeams).astype(np.float)
        signal = str(args.pol).strip()
        nside = int(args.nside_out)
        lmax_bnd = int(args.lmax_out)
        bias = float(args.ilc_bias)
        neighbors = int(args.neighborhood)
        if str(args.first_scale).strip() == 'fullsky' :
            k_min = 1
            kfirst = 0
        else :
            kfirst = min([int(args.first_scale),1])
        nlw = int(args.last_window)
        aic = int(args.aic_mod)
        write_cov = ndl.str2bool(str(args.write_covariance).strip())
        doGNILC = ndl.str2bool(str(args.do_foregrounds).strip())
        ssp = ndl.str2bool(str(args.project_fg).strip())
        weightsDir = str(args.weights_dir).strip()
        if len(weightsDir) :
            do_weight = False
            if weightsDir[-1] != '/' :
                weightsDir = weightsDir+'/'
            read_weight = True
        smooth = float(args.smooth_beam)
        masque = ndl.str2bool(str(args.masking).strip())
        mask_file = str(args.mask_file).strip()
        mask_size = float(args.mask_size)
        transition = float(args.mask_transition)
        ludwig = ndl.str2bool(str(args.do_ludwig).strip())
        ludwig_maps = str(args.ludwig_maps)
        ludwig_maps = ludwig_maps.split(',')
        ludwig_maps = list(map(str.strip, ludwig_maps))
        ludwig_maps = np.array(ludwig_maps)
        if len(ludwig_maps) > 1 :
            overwrite_ludwig = False
        ludwig_noises = str(args.ludwig_noises)
        ludwig_noises = ludwig_noises.split(',')
        ludwig_noises = list(map(str.strip, ludwig_noises))
        ludwig_noises = np.array(ludwig_noises)
        if len(ludwig_noises) > 1:
            overwrite_noise = False
        cmbs = str(args.cmbs)
        cmbs = cmbs.split(',')
        cmbs = list(map(str.strip, cmbs))
        cmbs = np.array(cmbs)
        if len(cmbs) > 1 :
            add_cmb = True
        firbs = str(args.firbs)
        firbs = firbs.split(',')
        firbs = list(map(str.strip, firbs))
        firbs = np.array(firbs)
        if len(firbs) > 1 :
            add_cib = True
        pfrac = float(args.pfrac)

        if os.path.exists(prior_cl.strip()) :
            cl_pr = np.loadtxt(prior_cl.strip())
            cl_pr = cl_pr.T[:,:3*nside]
            use_prior = True
        elif os.path.isfile(infold + prior_cl.strip()) :
            cl_pr = np.loadtxt(infold + prior_cl.strip())
            cl_pr = cl_pr.T[:,:3*nside]
            use_prior = True
        else :
            use_prior = False

        if os.path.exists(transfer_function.strip()) :
            F_l = np.loadtxt(transfer_function.strip())
            F_l = F_l.T[:,:3*nside]
            use_fl = True
        elif os.path.isfile(infold + transfer_function.strip()) :
            F_l = np.loadtxt(infold + transfer_function.strip())
            F_l = F_l.T[:,:3*nside]
            use_fl = True
        else :
            F_l = 1.
            use_fl = False


        print( "\n##############################################" )
        if masque :
            print( "\nNeedlet ILC Implementation (Masked) !\n" )
        else :
            print( "\nNeedlet ILC Implementation (Full-Sky) !\n" )
        print( "##############################################\n" )
        sys.stdout.flush()

        if ludwig :
            doGNILC = True

        if ludwig and overwrite_ludwig :
            ludwig_maps = inmaps
            print( "Ludwig maps set to input maps !" )

        if ludwig and overwrite_noise :
            ludwig_noises = noises
            print( "Ludwig noises set to input noises !" )

        if ludwig and overwrite_beam :
            lbeams = beams
            print( "Ludwig beams set to input beams !" )

        if (len(inmaps) != len(noises)) and doGNILC and do_weight :
            print( "Number of Input Maps Must Be Equal to Number of Input Noises !!!" )
            print( 'Quitting !!!' )
            sys.exit()

        if bl_in  :
            if len(inmaps) != len(bls) :
                print( "Number of Input Maps Must Be Equal to Number of Input Beams Files !!!" )
                print( 'Quitting !!!' )
                sys.exit()
        elif len(inmaps) != len(beams) :
            print( "Number of Input Maps Must Be Equal to Number of Input Beams !!!" )
            print( 'Quitting !!!' )
            sys.exit()

        if ludwig and (len(ludwig_maps) != len(ludwig_noises)) :
            print( "Number of Ludwig Maps Must Be Equal to Number of Ludwig Noises !!!" )
            print( 'Quitting !!!' )
            sys.exit()

        if ludwig and (len(ludwig_maps) != len(lbeams)) :
            print( "Number of Ludwig Maps Must Be Equal to Number of Ludwig Beams !!!" )
            print( 'Quitting !!!')
            sys.exit()

        if doGNILC :
            print( "\nForegrounds are Estimated by GNILC !!!" )
            if ssp :
                print( "Foregrounds Orthogonal to CMB - project_fg Set to True !!!" )
            else :
                print( "CMB Treated as Noise - project_fg Set to False !!!" )
                if not use_prior :
                    print( "\nA Prior on CMB Needed !!!" )
                    print( 'Quitting !!!')
                    sys.exit()
            if use_prior :
                print( "    Prior CMB Spectrum Read from \'"+prior_cl+"\'" )
                print( "    Transfer Function Read from \'"+transfer_function+"\'" )
                if not use_fl :
                    print( "    Transfer Function Set to Fl = 1. " )
        else :
            print( "\nForeground Mode is Off!!!" )

        sgnl = np.array(['I', 'Q', 'U'])

        ix, = np.where(sgnl == signal[0])
        nfirst = ix[0]

        ix, = np.where(sgnl == signal[-1])
        npol = ix[0] + 1

        if not os.path.exists(workfold) :
            os.mkdir(workfold)

        argparse_dict = vars(args)

        shutil.copy(confile, workfold+'gnilc_params.ini')

        with open(workfold+'gnilc_params.json', 'w') as outfile:
            json.dump(argparse_dict, outfile, indent=4)

        print( "\nOutput Directory = \'"+workfold+"\'" )
        if smooth :
            print( "\nSmoothing = "+str(smooth)+" arcmin" )
        if read_weight :
            print( "\nGNILC Weights are Read from : \'" + weightsDir +"\'")

        if not do_weight and not os.path.exists(weightsDir) :
            print( "\nWeights Directory Does Not Exist !!!" )
            print( 'Quitting !!!' )
            sys.exit()

        if gauss_bands :
            ell_bands, cntrs = ndl.gauss_bands(band_res, lmax_bnd)

        needletroot = workfold+'needlets'
        noisletroot = workfold+'noislets'
        outletroot = workfold+'outlets'
        bandroot = workfold+'ell_bands'
        synbandroot = workfold+'syn_ell_bands'

        needlist = [[],[],[]]
        noislist = [[],[],[]]
        outlist  = [[],[],[]]
        cmbout  = [[],[],[]]
        fgout  = [[],[],[]]
        biasout  = [[],[],[]]

        bandlist = []
        synbandlist = []

        npix = hp.nside2npix(nside)

        if bl_in :
            lm_s = np.zeros(len(bls))
            for b in range(len(bls)) :
                if os.path.exists(bls[b].strip()) :
                    bm_in = np.load(bls[b].strip())
                elif os.path.isfile(infold + bls[b].strip()) :
                    bm_in = np.load(infold + bls[b].strip())
                else :
                    print( "\nBeam File \'"+bls[b].strip()+"\' Does Not Exist !!!" )
                    print( 'Quitting !!!' )
                    sys.exit()

                if len(bm_in.shape) > 1 :
                    bm_in = bm_in[0]
                wh, = np.where(bm_in >= bl_min)
                lm_s[b] = max(wh)
        else :
            lm_s = np.zeros(len(beams))
            for b, bval in enumerate(beams) :
                bmg = hp.gauss_beam(bval*np.pi/180./60., lmax=8000)
                wh, = np.where(bmg >= bl_min)
                lm_s[b] = max(wh)
        lm_s = lm_s.astype(int)

        if masque :
            if len(mask_file) :
                print( "\nMask File = \'"+mask_file+"\'" )
                if os.path.exists(mask_file) :
                    mask = hp.read_map(mask_file, verbose=False)
                elif os.path.isfile(infold+mask_file) :
                    mask = hp.read_map(infold+mask_file, verbose=False)
                else :
                    print( "\nMask File Does Not Exist !!!" )
                    print( 'Quitting !!!' )
                    sys.exit()
                mask = hp.read_map(mask_file, verbose=False)
                mask = hp.ud_grade(mask, nside, pess=False, dtype=np.float64)
            else :
                print( "\nMasking with "+mask_size+"+"+transition+" deg Galactic Cosine Transition Mask" )
                mask = ndl.cos_transition_mask(mask_size, transition, nside)

            wh0, = np.where((1 - np.isfinite(mask)) + (mask == hp.UNSEEN))
            mask[wh0] = 0

            maske = np.copy(mask)
            wh0, = np.where(maske > 0)
            maske[wh0] = 1
            wh0, = np.where(maske == 0)
            maske[wh0] = np.NaN

        if not os.path.exists(workfold+'needlets.npy') :

            if use_prior :
                spec_pr = np.copy(cl_pr)
                lmn = cl_pr.shape[-1]
                if use_fl :
                    lmn = min([cl_pr.shape[-1], F_l.shape[-1]])
                    spec_pr = cl_pr[:,:lmn] * F_l[:,:lmn]
                nside_cmb = ndl.lower_powerof2(lmn//2)
                cmb_pr = np.array(hp.synfast(spec_pr, nside_cmb, new=True, verbose=False))

            for j in range(len(dets)) :

                dm = dets[j].strip()

                ##Input Signal
                ##------------

                inmapfile = inmaps[j].strip()
                if not os.path.exists(inmapfile) :
                    inmapfile = infold + inmaps[j].strip()
                    if not os.path.isfile(inmapfile) :
                        print( "\nInput \'"+inmaps[j].strip()+"\' File Does Not Exist !!!" )
                        print( 'Quitting !!!' )
                        sys.exit()
                if doGNILC and do_weight :
                    inoisfile = noises[j].strip()
                    if not os.path.exists(inoisfile) :
                        inoisfile = noisefold + noises[j].strip()
                        if not os.path.isfile(inoisfile) :
                            print( "\nInput \'"+noises[j].strip()+"\' File Does Not Exist !!!" )
                            print( 'Quitting !!!' )
                            sys.exit()
                    if add_cmb :
                        cmbfile = cmbs[j].strip()
                        if not os.path.exists(cmbfile) :
                            cmbfile = noisefold + cmbs[j].strip()
                            if not os.path.isfile(cmbfile) :
                                print( "\nInput CMB File Does Not Exist !!!" )
                                print( 'Quitting !!!' )
                                sys.exit()
                    if add_cib :
                        firbfile = firbs[j].strip()
                        if not os.path.exists(firbfile) :
                            firbfile = noisefold + firbs[j].strip()
                            if not os.path.isfile(firbfile) :
                                print( "\nInput FIRB File Does Not Exist !!!" )
                                print( 'Quitting !!!' )
                                sys.exit()

                if smooth :
                    b_out = smooth
                    bmg = hp.gauss_beam(smooth*np.pi/180./60., lmax=8000)
                    wh, = np.where(bmg >= bl_min)
                    lm_s[j] = max(wh)
                else :
                    b_out = 0

                if lm_s[j] > 2*nside :
                    lm_s[j] = 2*nside
                if lm_s[j] > np.max(cntrs) :
                    lm_s[j] = np.max(cntrs)
                wh, = np.where(cntrs < lm_s[j])
                jmx = np.max(wh)
                lm_s[j] = cntrs[jmx+1]

                bandfile = bandroot + str(j).zfill(2) + '.npy'
                bandlist.append(bandfile)

                synbandfile = synbandroot + str(j).zfill(2) + '.npy'

                if gauss_bands :
                    ell_bands, cntrs = ndl.gauss_bands(band_res, lmax_bnd, ana_bandfile=bandfile, syn_bandfile=synbandfile, verbose=False)
                else :
                    ell_bands = ndl.cosine_bands(cntrs[:jmx+2], bndfile=bandfile)
                    synbandfile = bandfile

                synbandlist.append(synbandfile)

                if bl_in :
                    if os.path.exists(bls[j].strip()) :
                        b_in = np.load(bls[j].strip())
                    elif os.path.isfile(infold + bls[j].strip()) :
                        b_in = np.load(infold + bls[j].strip())
                    else :
                        print( "\nBeam File \'"+bls[j].strip()+"\' Does Not Exist !!!" )
                        print( 'Quitting !!!' )
                        sys.exit()
                    if len(b_in.shape) < 2 :
                        b_in = np.array([b_in, b_in, b_in])
                else :
                    b_in = 3*[beams[j]]

                for pol in range(nfirst,npol) :
                    maps = np.array(hp.read_map(inmapfile, field=(pol), verbose=False))
                    whunseen, = np.where((1-np.isfinite(maps))+(maps == hp.UNSEEN))
                    if len(whunseen) :
                        maps[whunseen] = 0
                    if doGNILC and do_weight :
                        nmap = np.array(hp.read_map(inoisfile, field=(pol), verbose=False))
                        whunseen, = np.where((1-np.isfinite(nmap))+(nmap == hp.UNSEEN))
                        if len(whunseen) :
                            nmap[whunseen] = 0

                    nside_in = hp.get_nside(maps)

                    if masque :
                        msk = hp.ud_grade(mask, nside_in, pess=False, dtype=np.float64)
                    else :
                        msk = 1.

                    alms = ndl.change_resolution(maps*msk, beam_in=b_in[pol], beam_out=b_out, l_max=lm_s[j])
                    if doGNILC and do_weight :
                        nlms = ndl.change_resolution(nmap*msk, beam_in=b_in[pol], beam_out=b_out, l_max=lm_s[j])
                        if use_prior :
                            clms = ndl.change_resolution(cmb_pr[pol], beam_in=0., beam_out=b_out, l_max=lm_s[j])
                            nlms += clms

                    print( '\n Calculating Stokes ' + sgnl[pol] + ' Needlet Coefficients for Channel ' + str(j) + '\n' )
                    sys.stdout.flush()

                    needfile = needletroot + '_' + sgnl[pol] + 'map' + str(j).zfill(2)
                    if doGNILC and do_weight :
                        noisfile = noisletroot + '_' + sgnl[pol] + 'map' + str(j).zfill(2)

                    maplist = ndl.alm2needlets(alms, needfile, ell_bands, needletnside=nside)
                    needlist[pol].append(maplist)

                    if doGNILC and do_weight :
                        maplist = ndl.alm2needlets(nlms, noisfile, ell_bands, needletnside=nside)
                        noislist[pol].append(maplist)

                    gc.collect()

            if ludwig :

                for b, bval in enumerate(lbeams) :
                    bmg = hp.gauss_beam(bval*np.pi/180./60., lmax=8000)
                    wh, = np.where(bmg >= bl_min)
                    lm_s[b] = max(wh)

                for j in range(len(ludwig_maps)) :

                    if lm_s[j] > 2*nside :
                        lm_s[j] = 2*nside
                        wh, = np.where(cntrs < lm_s[j])
                        jmx = np.max(wh)
                        lm_s[j] = cntrs[jmx+1]

                    ludmapfile = ludwig_maps[j].strip()
                    if not os.path.exists(ludmapfile) :
                        ludmapfile = infold + ludwig_maps[j].strip()

                    ludmap = hp.read_map(ludmapfile, verbose=False)
                    whunseen, = np.where((1-np.isfinite(ludmap))+(ludmap == hp.UNSEEN))
                    if len(whunseen) :
                        ludmap[whunseen] = 0

                    ludnoisfile = ludwig_noises[j].strip()
                    if not os.path.exists(ludnoisfile) :
                        ludnoisfile = noisefold + ludwig_noises[j].strip()

                    ludnoise = hp.read_map(ludnoisfile, verbose=False)
                    whunseen, = np.where((1-np.isfinite(ludnoise))+(ludnoise == hp.UNSEEN))
                    if len(whunseen) :
                        ludnoise[whunseen] = 0

                    lwg_maps = ndl.ludwig2qu(ludmap*pfrac)
                    lwg_nois = ndl.ludwig2qu(ludnoise*pfrac)

                    b_in = lbeams[j]
                    if smooth :
                        b_out = smooth
                    else :
                        b_out = 0

                    if masque :
                        nside_in = hp.get_nside(lwg_maps)
                        msk = hp.ud_grade(mask, nside_in, pess=False, dtype=np.float64)
                    else :
                        msk = 1.

                    lwg1_lms = ndl.change_resolution(lwg_maps[0]*msk, beam_in=b_in, beam_out=b_out, l_max=lm_s[j])
                    lwg2_lms = ndl.change_resolution(lwg_maps[1]*msk, beam_in=b_in, beam_out=b_out, l_max=lm_s[j])

                    noise_lwg1 = ndl.change_resolution(lwg_nois[0]*msk, beam_in=b_in, beam_out=b_out, l_max=lm_s[j])
                    noise_lwg2 = ndl.change_resolution(lwg_nois[1]*msk, beam_in=b_in, beam_out=b_out, l_max=lm_s[j])

                    if overwrite_noise :
                        cmap = 0.
                        if add_cmb :
                            cmbfile = noisefold + cmbs[j].strip()
                            cmap = np.array(hp.read_map(cmbfile, verbose=False))
                        if add_cib :
                            firbfile = noisefold + firbs[j].strip()
                            fmap = np.array(hp.read_map(firbfile, verbose=False))
                            cmap += fmap
                        if add_cmb or add_cib :
                            lwg_cmb = ndl.ludwig2qu(cmap*pfrac)
                            cmb_lwg1 = ndl.change_resolution(lwg_cmb[0]*msk, beam_in=b_in, beam_out=b_out, l_max=lm_s[j])
                            cmb_lwg2 = ndl.change_resolution(lwg_cmb[1]*msk, beam_in=b_in, beam_out=b_out, l_max=lm_s[j])
                            noise_lwg1 += cmb_lwg1
                            noise_lwg2 += cmb_lwg2

                    print( '\n Calculating Ludwig Needlet Coefficients for Channel '+str(2*j+len(dets))+' and '+str(2*j+1+len(dets))+'\n' )
                    sys.stdout.flush()

                    for pol in range(1,npol) :

                        need1file = needletroot + '_ludwig1_' + sgnl[pol] + 'map' + str(j).zfill(2)
                        maplist = ndl.alm2needlets(lwg1_lms[pol], need1file, ell_bands, needletnside=nside)
                        needlist[pol].append(maplist)

                        need2file = needletroot + '_ludwig2_' + sgnl[pol] + 'map' + str(j).zfill(2)
                        maplist = ndl.alm2needlets(lwg2_lms[pol], need2file, ell_bands, needletnside=nside)
                        needlist[pol].append(maplist)

                        nois1file = noisletroot + '_ludwig1_' + sgnl[pol] + 'map' + str(j).zfill(2)
                        maplist = ndl.alm2needlets(noise_lwg1[i], nois1file, ell_bands, needletnside=nside)
                        noislist[i].append(maplist)

                        nois2file = noisletroot + '_ludwig2_' + sgnl[pol] + 'map' + str(j).zfill(2)
                        maplist = ndl.alm2needlets(noise_lwg2[pol], nois2file, ell_bands, needletnside=nside)
                        noislist[pol].append(maplist)

                    gc.collect()


            print("\nNeedlets are done in "+str(time.time() - start)+" seconds\n")

            needlist = np.array(needlist)
            bandlist = np.array(bandlist)
            synbandlist = np.array(synbandlist)

            np.save(workfold+'needlets.npy', needlist)
            np.save(workfold+'bandlist.npy', bandlist)
            np.save(workfold+'synbandlist.npy', synbandlist)

            if doGNILC and do_weight :
                noislist = np.array(noislist)
                np.save(workfold+'noislets.npy', noislist)


        needlist = np.load(workfold+'needlets.npy')
        bandlist = np.load(workfold+'bandlist.npy')
        synbandlist = np.load(workfold+'synbandlist.npy')
        if doGNILC and do_weight :
            noislist = np.load(workfold+'noislets.npy')

        for pol in range(nfirst,npol) :

            ell_bands = np.load(bandlist[-1])  # ell bands at last channel

            boylar = np.array(list(map(len, needlist[pol])))
            k_max = np.max(boylar)   # max number of scales across all channels

            for k in range(kfirst, k_max-nlw) : # for each scale

                print( '\n Calculating '+sgnl[pol]+' Covariance Matrices at Scale ' + str(k) )

                if ludwig and k >= 1 :
                    ch, = np.where(boylar > k)
                else :
                    ch, = np.where(boylar[0:len(dets)] > k)
                boy = len(ch)

                print('  '+str(boy)+' Channels at Scale '+str(k))
                sys.stdout.flush()

                ## To find the size of the covariance domain n.pix(R) ;
                ## use ilc-bias = (n.chans - 1) / [ (n.pix(R) / n.tot) * n.ell-modes ]
                ## where n.ell-modes = ((lmax+1)**2 - lmin**2) and n.tot = 12*nside**2
                lmx=np.max(np.where(ell_bands[k] != 0))
                lmn=np.min(np.where(ell_bands[k] != 0))
                if gauss_bands :
                    lmx=np.max(np.where(ell_bands[k] >= bl_min))
                    lmn=np.min(np.where(ell_bands[k] >= bl_min))
                f1 = np.sqrt((boy-1)/bias)
                fs = np.sqrt((lmx+1)**2-lmn**2)

                ## No. of nearest neighbors to include in the covariance domain :
                if cntrs[k] < 80 :
                    mx_near = max([2,neighbors])
                else :
                    mx_near = neighbors

                ## Degrade to speed up covariance calculations :
                dgrade = ndl.lower_powerof2( np.sqrt(12)*nside * f1/fs / (2*mx_near+1.))
                if dgrade > nside//4 :
                    mudi = nside//dgrade
                    dgrade = nside//4
                    mx_near += 1
                    print('  Covariance nside = '+str(nside//dgrade)+' (rescaled from '+str(mudi)+')')
                elif dgrade < nside//32 :
                    mudi = nside//dgrade
                    dgrade = nside//32
                    mx_near = max([mx_near-1,1])
                    print('  Covariance nside = '+str(nside//dgrade)+' (rescaled from '+str(mudi)+')')
                else :
                    print('  Covariance nside = '+str(nside//dgrade))
                sys.stdout.flush()

                n_modes = (dgrade*(2*mx_near+1))**2
                if k < k_min :
                    n_modes = hp.nside2npix(nside)
                    if masque :
                        wh0, = np.where(np.isfinite(maske)*(maske != hp.UNSEEN))
                        n_modes = len(wh0)

                R1m = np.array(np.zeros((boy, boy), dtype=np.float64))
                N1m = np.array(np.zeros((boy, boy), dtype=np.float64))

                Rij = []
                Nij = []
                yvec = []

                fgnilc = []
                cmbilc = []

                mapk = np.zeros((boy,npix))
                cmbk = np.zeros((boy,npix))

                ## Bias term :
                mapb = np.zeros((boy,npix))

                dimfile = workfold+'dimension_'+ sgnl[pol] + 'map' +str(k).zfill(2)+'.fits'
                ilcfgfile = workfold+'fgnilc_'+ sgnl[pol] + 'weights' +str(k).zfill(2)+'.npy'
                ilcmbfile = workfold+'cmbilc_'+ sgnl[pol] + 'weights' +str(k).zfill(2)+'.npy'

                if not os.path.exists(dimfile) :
                    for i in range(boy) :
                        Rmap1 = hp.read_map(needlist[pol][ch[i]][k], verbose=False)
                        if doGNILC and do_weight :
                            Nmap1 = hp.read_map(noislist[pol][ch[i]][k], verbose=False)
                        yvec.append(Rmap1)
                        for j in range(i, boy) :
                            Rmap2 = hp.read_map(needlist[pol][ch[j]][k], verbose=False)
                            if doGNILC and do_weight :
                                Nmap2 = hp.read_map(noislist[pol][ch[j]][k], verbose=False)
                            if k < k_min :
                                if read_weight :
                                    covfile = weightsDir+'cov'+sgnl[pol]+str(k)+'_'+str(ch[i])+'x'+str(ch[j])+'.fits'
                                else :
                                    covfile = workfold+'cov'+sgnl[pol]+str(k)+'_'+str(ch[i])+'x'+str(ch[j])+'.fits'

                                if doGNILC and do_weight :
                                    novfile = workfold+'nov'+sgnl[pol]+str(k)+'_'+str(ch[i])+'x'+str(ch[j])+'.fits'

                                if not os.path.exists(covfile) :
                                    msq = 1.
                                    if masque :
                                        msq = mask
                                    mcov = Rmap1*Rmap2*msq
                                    mdx, = np.where(np.isfinite(mcov)*(mcov != hp.UNSEEN)*(msq != 0))
                                    rcov = np.mean(mcov[mdx])
                                    rvoc = hp.UNSEEN*np.ones(hp.nside2npix(nside))
                                    rvoc[mdx] = 1
                                    relt = hp.ud_grade(rvoc, nside//dgrade)
                                    ndx, = np.where(relt != hp.UNSEEN)
                                    relt[ndx] = rcov
                                    if doGNILC and do_weight :
                                        mcov = Nmap1*Nmap2*msq
                                        mdx, = np.where(np.isfinite(mcov)*(mcov != hp.UNSEEN)*(msq != 0))
                                        ncov = np.mean(mcov[mdx])
                                        nvoc = hp.UNSEEN*np.ones(hp.nside2npix(nside))
                                        nvoc[mdx] = 1
                                        nelt = hp.ud_grade(nvoc, nside//dgrade)
                                        ndx, = np.where(nelt != hp.UNSEEN)
                                        nelt[ndx] = ncov

                                    if write_cov :
                                        hp.write_map(covfile, relt, overwrite=True)
                                        if doGNILC and do_weight :
                                            if write_cov :
                                                hp.write_map(novfile, nelt, overwrite=True)

                                if os.path.exists(covfile) :
                                    relt = hp.read_map(covfile, verbose=False)

                                if doGNILC and do_weight :
                                    if os.path.exists(novfile) :
                                        nelt = hp.read_map(novfile, verbose=False)

                            else :
                                if read_weight :
                                    covfile = weightsDir+'cov'+sgnl[pol]+str(k)+'_'+str(ch[i])+'x'+str(ch[j])+'.fits'
                                else :
                                    covfile = workfold+'cov'+sgnl[pol]+str(k)+'_'+str(ch[i])+'x'+str(ch[j])+'.fits'

                                if doGNILC and do_weight :
                                    novfile = workfold+'nov'+sgnl[pol]+str(k)+'_'+str(ch[i])+'x'+str(ch[j])+'.fits'

                                if not os.path.exists(covfile) :
                                    if masque :
                                        relt = ndl.local_cov(Rmap1, Rmap2, mask=mask, degrade=dgrade, neigh_size=mx_near)
                                    else :
                                        relt = ndl.local_cov(Rmap1, Rmap2, degrade=dgrade, neigh_size=mx_near)
                                    if write_cov :
                                        hp.write_map(covfile, relt, overwrite=True)
                                if os.path.exists(covfile) :
                                    relt = hp.read_map(covfile, verbose=False)

                                if doGNILC and do_weight :
                                    if not os.path.exists(novfile) :
                                        if masque :
                                            nelt = ndl.local_cov(Nmap1, Nmap2, mask=mask, degrade=dgrade, neigh_size=mx_near)
                                        else :
                                            nelt = ndl.local_cov(Nmap1, Nmap2, degrade=dgrade, neigh_size=mx_near)
                                        if write_cov :
                                            hp.write_map(novfile, nelt, overwrite=True)
                                    if os.path.exists(novfile) :
                                        nelt = hp.read_map(novfile, verbose=False)

                            Rij.append(relt)
                            if doGNILC and do_weight :
                                Nij.append(nelt)

                    Rij = np.array(Rij)
                    yvec = np.array(yvec)
                    if doGNILC and do_weight :
                        Nij = np.array(Nij)

                    lngth = boy
                    lgcmb = boy

                    dmap = np.zeros(len(yvec[0]))
                    if read_weight :
                        filein = weightsDir+'dimension_'+ sgnl[pol] + 'map' +str(k).zfill(2)+'.fits'
                        dmap = hp.read_map(filein, verbose=False)
                        filein = weightsDir+'cmbilc_'+ sgnl[pol] + 'weights' +str(k).zfill(2)+'.npy'
                        cmb_weights = np.load(filein)
                        if doGNILC :
                            filein = weightsDir+'fgnilc_'+ sgnl[pol] + 'weights' +str(k).zfill(2)+'.npy'
                            fg_weights = np.load(filein)

                    print( '  Calculating NILC Filters for '+sgnl[pol]+' at Scale ' + str(k) )
                    sys.stdout.flush()

                    zort = 0
                    zonk = 0
                    bonk = 0

                    bort = []

                    whseen, = np.where(np.isfinite(Rij[0]) * (Rij[0] != hp.UNSEEN) * (Rij[0] != 0))

                    fgnilc.append(whseen)
                    cmbilc.append(whseen)

                    for px in whseen : # at each superpixel

                        if k < k_min :
                            indx = np.arange(npix)
                        else :
                            indx = ndl.pix_in_pix(hp.get_nside(Rij[0]), nside, px)

                        cnt = 0
                        for i in range(boy) :
                            for j in range(i, boy) :
                                R1m[i,j] = Rij[cnt, px]
                                if doGNILC and do_weight :
                                    N1m[i,j] = Nij[cnt, px]
                                cnt += 1
                                if j != i :
                                    R1m[j,i] = R1m[i,j]
                                    if doGNILC and do_weight :
                                        N1m[j,i] = N1m[i,j]

                        R1I = lng.pinv(R1m)    ## R^{-1}

                        ## CMB ILC Filter
                        if read_weight :
                            wheq, = np.where(cmb_weights[0] == px)
                            if len(wheq) == 0 :
                                print( 'Weights Must Match the Analysed Map Region!!' )
                                print( 'Quitting!!' )
                                sys.exit()
                            else :
                                Vvec = cmb_weights[wheq[0]+1]
                        else :
                            Vvec = np.ones(len(R1I))
                            Amat = np.outer(Vvec,Vvec)
                            N2I = Vvec.dot(R1I).dot(Vvec)

                            Vvec = Amat.dot(R1I)
                            if N2I != 0. :
                                Vvec /= N2I

                        x = Vvec.dot(yvec[:,indx])  ## Filtering within the superpixel

                        cmbilc.append(Vvec)

                        uzn = len(x)

                        if uzn < lgcmb :
                            cmbk = cmbk[0:uzn]
                            cmbk[:,indx] = x
                        else :
                            cmbk[:,indx] = x[0:lgcmb]
                        lgcmb = len(cmbk)

                        Rmat = np.copy(R1m)

                        if doGNILC :
                            if do_weight :

                                u, s, vh = lng.svd(N1m)

                                u = (u + vh.T)/2.

                                wh, = np.where(s > 0.)

                                dims = len(s)
                                dim  = len(s)

                                if len(wh) == 0 :
                                    if zort == 0 :
                                        print( '   ALL EIGENVALES ARE NEGATIVE !!!' )
                                        sys.stdout.flush()
                                    s = np.abs(s)
                                    wh = range(len(s))

                                u = u[:,wh]
                                s = np.sqrt(s[wh])

                                N2mI = (u/s).dot(u.T)   ## N^{-1/2}
                                N2mI = (N2mI + N2mI.T)/2  ## Symmetrize !!!

                                Mmat = N2mI.dot(Rmat).dot(N2mI)

                                Mmat = (Mmat + Mmat.T)/2  ## Symmetrize !!!

                                N2I = (u*s).dot(u.T)   ## N^{1/2}
                                N2I = (N2I + N2I.T)/2  ## Symmetrize !!!

                                u,s,vh = lng.svd(Mmat)

                                u = (u+vh.T)/2.

                                dims = len(s)

                                if (np.max(s) <= 1.) and (zonk == 0) :
                                    print( '   WARNING!!! ALL EIGENVALES ARE SMALLER THAN THE NOISE COVARIANCE!!!' )
                                    sys.stdout.flush()
                                    zonk +=1

                                dim = len(s)

                                if dim != 0 :
                                    if aic == 0 or aic == 1 :
                                        dim = ndl.num_dof(s, n_modes)
                                    elif aic == 2 :
                                        dim = ndl.num_dof(s, n_modes, use_mdl=True)
                                    else :
                                        dim = ndl.akaike(s)

                                wh = range(dim)

                                wh, = np.where(s[wh] > 1.)
                                dim = len(wh)

                                bort.append(dims - dim)
                                dmap[indx] = dim

                            else:
                                dim = dmap[indx[0]]

                            if (dim == 0) or (dim == 1 and ssp) :

                                bonk += 1
                                mapk[:,indx] = 0.
                                mapb[:,indx] = 0.

                                fgnilc.append(None)

                            else :

                                if do_weight :

                                    s = s[wh]-1.
                                    u = u[:,wh]

                                    if ssp :
                                        wn, = np.where(s <= 0.)
                                        if len(wn) :
                                            if zort == 0 :
                                                print('WARNING!!! '+str(len(wn))+' Negative Diagonals Set to 1.e-12')
                                                zort += 1
                                            s[wn] = 1.e-12
                                        s = np.sqrt(s)

                                        Mmat = np.dot(N2I,u)*s

                                        ## Project CMB onto signal-space (1,1,1..) -> a (a-tilde):
                                        atilde = lng.pinv(np.dot(Mmat.T,Mmat))
                                        atilde = atilde.dot(np.dot(Mmat.T, np.ones(dims)))
                                        if np.sum(atilde**2) != 0. :
                                            atilde /= np.sqrt(np.sum(atilde**2))

                                        ## Projection onto foreground subspace :
                                        if aic == 0 :
                                            utilde = np.dot(Mmat, atilde)
                                            utilde = ndl.proj_MorthV(u, utilde)
                                            Vmat = np.dot(N2I,utilde)
                                        else :
                                            ## Build rotation matrix P such that P[0] = a :
                                            Pmat = ndl.matrix_SOn(atilde)

                                            ## Find directions orthogonal to CMB :
                                            Pmat = Pmat[:,1:]

                                            ## Basis of the foreground subspace :
                                            Vmat = np.dot(Mmat,Pmat)
                                        Vvec = np.dot(Mmat,atilde)
                                    else :
                                        Vmat = np.dot(N2I,u)

                                else :
                                    wheq, = np.where(fg_weights[0] == px)
                                    if len(wheq) == 0 :
                                        print( 'Weights Must Match the Analysed Map Region!!')
                                        print( 'Quitting!!')
                                        sys.exit()
                                    else :
                                        Vmat = fg_weights[wheq[0]+1]

                                if read_weight :
                                    wheq, = np.where(fg_weights[0] == px)
                                    if len(wheq) == 0 :
                                        print( 'Weights Must Match the Analysed Map Region!!')
                                        print( 'Quitting!!')
                                        sys.exit()
                                    else :
                                        Vmat = fg_weights[wheq[0]+1]
                                else :
                                    Amat = np.dot(Vmat.T,R1I)

                                    N2I = lng.pinv(Amat.dot(Vmat))
                                    Vmat = Vmat.dot(N2I).dot(Amat)  ## ILC Filter

                                fgnilc.append(Vmat)

                                x = Vmat.dot(yvec[:,indx])  ## Filtering within the superpixel

                                fgb = Vmat.dot(cmbk[:,indx])  ## Bias within the superpixel

                                x = x - fgb

                                uzn = len(x)

                                if uzn < lngth :
                                    mapk = mapk[0:uzn]
                                    mapk[:,indx] = x

                                    mapb = mapb[0:uzn]
                                    mapb[:,indx] = fgb
                                else :
                                    mapk[:,indx] = x[0:lngth]

                                    mapb[:,indx] = fgb[0:lngth]

                                lngth = len(mapk)

                            zort = 1

                    if doGNILC :
                        if do_weight :
                            print( '   Discarding ' + str(np.max(bort)) + ' Eigenvalue(s)')
                            sys.stdout.flush()

                        if do_weight and bonk != 0 :
                            print( '   Warning: All Eigenvalues Discarded in ' + str(bonk) +' Superpixels')
                            sys.stdout.flush()

                        if not read_weight :
                            fgnilc = np.array(fgnilc)
                            np.save(ilcfgfile, fgnilc)

                    hp.write_map(dimfile, dmap, overwrite=True)

                    if not read_weight :
                        cmbilc = np.array(cmbilc)
                        np.save(ilcmbfile, cmbilc)

                j = 0
                ndlcmb = []
                foregd = []
                fgbias = []
                for i in range(len(dets)) :
                    if i in ch :
                        cmbfile = outletroot + '_' + sgnl[pol] + 'cmb' + str(i).zfill(2) + '_scale' + str(k).zfill(2) + '.fits'
                        fgnilcfile = outletroot + '_' + sgnl[pol] + 'fg_gnilc' + str(i).zfill(2) + '_scale' + str(k).zfill(2) + '.fits'
                        fgbiasfile = outletroot + '_' + sgnl[pol] + 'fg_bias' + str(i).zfill(2) + '_scale' + str(k).zfill(2) + '.fits'
                        if os.path.exists(cmbfile) :
                            print( '  Filtered '+sgnl[pol]+' Maps Already Exist for Channel ' + dets[i] )
                        else :
                            hp.write_map(cmbfile, cmbk[j], overwrite=True)
                            if masque :
                                nside_in = hp.get_nside(cmbk[j])
                                msk = hp.ud_grade(mask, nside_in, pess=False, dtype=np.float64)
                            else :
                                msk = 1.
                            if doGNILC :
                                hp.write_map(fgnilcfile, mapk[j]*msk, overwrite=True)
                                hp.write_map(fgbiasfile, mapb[j]*msk, overwrite=True)
                        j += 1
                    else :
                        cmbfile = ''
                        fgnilcfile = ''
                        fgbiasfile = ''
                    ndlcmb.append(cmbfile)
                    foregd.append(fgnilcfile)
                    fgbias.append(fgbiasfile)

                cmbout[pol].append(ndlcmb)
                fgout[pol].append(foregd)
                biasout[pol].append(fgbias)

            np.save(workfold+'cmbout'+sgnl[pol]+'.npy', cmbout[pol])
            if doGNILC :
                np.save(workfold+'fgnilcout'+sgnl[pol]+'.npy', fgout[pol])
                np.save(workfold+'fgbiasout'+sgnl[pol]+'.npy', biasout[pol])

        if os.path.exists(workfold+'fgnilcoutI.npy') or os.path.exists(workfold+'cmboutI.npy'):
            nfirst = 0
        elif os.path.exists(workfold+'fgnilcoutQ.npy') or os.path.exists(workfold+'cmboutQ.npy'):
            nfirst = 1
        else :
            nfirst = 2

        print('\n')

        for pol in range(nfirst, npol) :
            cmbout[pol] = np.load(workfold+'cmbout'+sgnl[pol]+'.npy')
            if doGNILC :
                fgout[pol] = np.load(workfold+'fgnilcout'+sgnl[pol]+'.npy')
                biasout[pol] = np.load(workfold+'fgbiasout'+sgnl[pol]+'.npy')

        cmbout  = np.array(cmbout)
        fgout  = np.array(fgout)
        biasout  = np.array(biasout)

        if masque :
            wh0, = np.where((1 - np.isfinite(maske))+(maske == hp.UNSEEN))
            mask_nside = hp.get_nside(maske)
            this_nside = mask_nside

        jmax = len(cmbout[nfirst][0])
        for j in range(jmax) :

            print( ' Constructing the CMB Component at Channel ' + dets[j] )
            sys.stdout.flush()
            if doGNILC :
                print( ' Constructing the Total Foreground Component at Channel ' + dets[j] )
                sys.stdout.flush()

            ell_bands = np.load(synbandlist[j])  # ell bands at channel j

            b_max = len(ell_bands)

            l_max = len(ell_bands[kfirst]) - 1

            if smooth :
                finside = nside
            else :
                maps = hp.read_map(infold + inmaps[j].strip(), verbose=False)
                finside = hp.get_nside(maps)

            last_wndw = 0

            if bl_in :
                if os.path.exists(bls[j].strip()) :
                    b_o = np.load(bls[j].strip())
                else :
                    b_o = np.load(infold + bls[j].strip())
                if gauss_bands and nlw :
                    if np.ndim(b_o) > 1 :
                        whlmx, = np.where(b_o[0] >= bl_min)
                    else :
                        whlmx, = np.where(b_o >= bl_min)
                    l_mx = max(whlmx)
                    b_chan = ndl.get_resolution(l_mx, bl_min)
                    whbm, = np.where(band_res[:len(band_res)-nlw+1] >= b_chan)
                    b_chan = band_res[max(whbm)]
                    last_wndw = len(band_res)-len(whbm)
            else :
                b_o = 3*[beams[j]]
                if gauss_bands and nlw :
                    whbm, = np.where(band_res[:len(band_res)-nlw+1] >= beams[j])
                    b_chan = band_res[max(whbm)]
                    last_wndw = len(band_res)-len(whbm)

            if gauss_bands and nlw :
                bl_ch = hp.gauss_beam(b_chan*np.pi/180./60., lmax=l_max)

            cmbmap = []
            foremap = []
            biasmap = []

            finpix = hp.nside2npix(finside)
            m0 = np.zeros(finpix)

            if j == jmax-1 :
                fullres_cmb = np.zeros((3,finpix))

            for pol in range(nfirst) :
                cmbmap.append(m0)
                if doGNILC :
                    foremap.append(m0)
                    biasmap.append(m0)

            for pol in range(nfirst, npol) :
                cmb_list = [cmbout[pol][i][j] for i in range(b_max-kfirst-nlw)]
                alms_cmb = ndl.needlets2alm(cmb_list, ell_bands[kfirst:b_max-nlw], nolastwindow=last_wndw)#, scale_map=sfile))
                if gauss_bands and nlw and smooth:
                    alms_cmb = hp.almxfl(alms_cmb, 1./bl_ch)
                cmbmap.append(hp.alm2map(alms_cmb, finside, verbose=False))
                if doGNILC :
                    fin_list = [fgout[pol][i][j] for i in range(b_max-kfirst-nlw)]
                    sfile = workfold+'fgnilcmap'+sgnl[pol]+str(j).zfill(2)
                    alms_fin = ndl.needlets2alm(fin_list, ell_bands[kfirst:b_max-nlw], nolastwindow=last_wndw)#, scale_map=sfile))
                    if gauss_bands and nlw and smooth:
                        alms_fin = hp.almxfl(alms_fin, 1./bl_ch)
                    foremap.append(hp.alm2map(alms_fin, finside, verbose=False))

                    bias_list = [biasout[pol][i][j] for i in range(b_max-kfirst-nlw)]
                    alms_bias = ndl.needlets2alm(bias_list, ell_bands[kfirst:b_max-nlw], nolastwindow=last_wndw)#, scale_map=sfile))
                    if gauss_bands and nlw and smooth:
                        alms_bias = hp.almxfl(alms_bias, 1./bl_ch)
                    biasmap.append(hp.alm2map(alms_bias, finside, verbose=False))

                if not smooth and not nlw :
                    if j == jmax-1 :
                        fullres_cmb[pol] = cmbmap[pol]
                    cmbmap[pol] = ndl.change_resolution(cmbmap[pol], map_out=True, beam_out=b_o[pol])
                    if doGNILC :
                        foremap[pol] = ndl.change_resolution(foremap[pol], map_out=True, beam_out=b_o[pol])
                        biasmap[pol] = ndl.change_resolution(biasmap[pol], map_out=True, beam_out=b_o[pol])

            for pol in range(npol,3) :
                cmbmap.append(m0)
                if doGNILC :
                    foremap.append(m0)
                    biasmap.append(m0)

            cmbmap = np.array(cmbmap)
            foremap = np.array(foremap)
            bcfgmap = np.array(biasmap)

            if masque :
                if finside == mask_nside :
                    wh1 = wh0
                    this_nside = finside
                elif finside != this_nside :
                    this_mask = hp.ud_grade(maske, finside)
                    wh1, = np.where((1-np.isfinite(this_mask))+(this_mask == hp.UNSEEN))
                    this_nside = finside
                for pol in range(3) :
                    cmbmap[pol,wh1] = hp.UNSEEN
                    if j == jmax-1 :
                        fullres_cmb[pol,wh1] = hp.UNSEEN
                    if doGNILC :
                        foremap[pol,wh1] = hp.UNSEEN
                        bcfgmap[pol,wh1] = hp.UNSEEN

            sufx = ''
            if gauss_bands and nlw :
                sufx = '_'+str(int(b_chan))+'arcmin'
            if smooth :
                sufx = '_'+str(int(smooth))+'arcmin'

            if doGNILC :
                forename = 'foreground_' + dets[j].strip() + sufx + '.fits'
                hp.write_map(workfold+forename, foremap, overwrite=True)
                bcfgname = 'bias_' + dets[j].strip() + sufx + '.fits'
                hp.write_map(workfold+bcfgname, bcfgmap, overwrite=True)
            hp.write_map(workfold+'cmb_' + dets[j].strip() + sufx + '.fits', cmbmap, overwrite=True)
            if not smooth and j == jmax-1 and not nlw :
                hp.write_map(workfold+'fullres_cmb.fits', fullres_cmb, overwrite=True)


        print("\nDONE in "+str(time.time() - start)+" seconds !!!\n")

        return(0)

if __name__ == "__main__":
    sys.exit(main())

