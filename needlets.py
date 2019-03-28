import sys
import numpy as np
import healpy as hp
import argparse
from numpy.linalg import pinv


### CONSTRUCT A GAUSSIAN BEAM OF SIZE FWHM
###---------------------------------------

def beam_profile(pixsize, npix, fwhm = 10, center=0) :
    x = np.arange(0, npix, 1, float)*pixsize
    y = x[:,np.newaxis]

    if np.isscalar(center) :
        x0 = y0 = npix*pixsize / 2
    else :
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)



### FIND NEIGHBOURS OF A HEALPIX PIXEL
###-----------------------------------

def hpx_neighbours(ipix, vicinity, nside, ordering='Ring') :


    neighbours = np.array([ipix])
    remote = np.array([0])

    nested = False
    if ordering is 'Nest' :
        nested = True

    if vicinity != 0 :

        for r in range(1,vicinity+1) :

            if r == 1 :
                liste = hp.get_all_neighbours(nside, ipix, nest=nested)
                liste = liste[np.where(liste >= 0)]
                remote = np.concatenate([remote, np.tile(r, liste.size)])
                neighbours = np.concatenate([neighbours, liste])
                previouslist = np.copy(liste)
            else :
                liste = hp.get_all_neighbours(nside, previouslist, nest=nested)
                liste = np.unique(liste)
                liste = liste[np.where(liste >= 0)]

                ## Eliminate those already in list
                previouslist = np.concatenate([neighbours, liste])

                liste, inds = np.unique(previouslist, return_inverse=True)

                indx=inds[0:len(neighbours)]

                previouslist = np.delete(liste, indx)

                neighbours = np.concatenate([liste[indx], previouslist])
                remote = np.concatenate([remote, np.tile(r, previouslist.size)])

    return (neighbours, remote)




### cos^2 Apodization Function
###---------------------------

def apod(len_tot, min_len, max_len) :

    apfac = np.ones(len_tot)

    for i in range(min_len, len_tot):

        if i <= max_len :
            apfac[i] = np.cos((np.pi/2.)*(i-min_len)/(max_len-min_len))**2
        else :
            apfac[i] = 0.

    return apfac



### CONSTRUCT LOCAL COVARIANCE MATRIX OF BACKGROUND EMISSION
###---------------------------------------------------------

def background_covariance(maps, nsidecov=0, ordering='RING', pixperscale=32, smooth=False) :

    print('Calculating Background Covariance')

    cov =[]
    if type(maps[0]) is not type(maps) :
        cov = cov + [maps*maps]
    else :
        for i in range(len(maps)) :
            for j in range(i, len(maps)) :
                cov = cov + [maps[i]*maps[j]]

    cov = np.array(cov)
    npix  = len(cov[0])
    nside = hp.npix2nside(npix)
    if nsidecov == 0 :
        nsidecov = nside

    rmat=[]
    for i in range(len(cov)) :
        if smooth is True :
            nside_out = nside//4
            stat = hp.ud_grade(cov[i], nside_out, pess=False, order_in=ordering, order_out=ordering)

            lmax = 2*nside_out
            alm = hp.map2alm(stat, lmax)

            pixsize = np.sqrt(4.*np.pi/npix)
            fwhm = pixperscale*pixsize

            stat = hp.alm2map(alm, nsidecov, fwhm=fwhm, inplace=True, verbose=False)
        else :
            nside_out = nside//pixperscale
            stat = hp.ud_grade(cov[i], nside_out, pess=False, order_in=ordering, order_out=ordering)
            stat = hp.ud_grade(stat, nside, pess=False, order_in=ordering, order_out=ordering)

        rmat = rmat + [stat]


    print('Background Covariance Calculated')

    return np.array(rmat)


### FIND MULTIPOLE BIN OF A PIXEL IN FOURIER DOMAIN
###------------------------------------------------

def getBin(i, j, npix, pixsize, dell) :

    dk = 180.*60./pixsize/npix/np.pi

    if i < npix // 2 :
        x = i*dk
    else:
        x = (i - npix)*dk
    if j < npix // 2 :
        y = j*dk
    else :
        y = (j - npix)*dk

    ell = np.pi * 2. * np.sqrt(x**2 + y**2)

    nbin = int(ell//dell)

    return nbin


### LIST OF PIXELS IN A SUPERPIXEL
###-------------------------------

def pix_in_pix(nside_low, nside_high, ipix, ord_in='Ring', ord_out='Ring') :

    rat = int(nside_high//nside_low)**2
    if ord_in is 'Ring' :
        ipix = hp.ring2nest(nside_low, ipix)
    ipix *= rat
    liste = np.arange(ipix,ipix+rat)
    if ord_out is 'Ring' :
        liste = hp.nest2ring(nside_high, liste)

    return liste




##FACE_PIX: function returns the pixel indices of a HealPix face
##---------------------------------------------------------------

def face_pix(nside, face=0, ordering='Ring') :

    while face > 11 :
        face -= 12

    if nside is 1 :
        subpix = np.array([face])
    else :
        tmpnside = 2

        subpix = np.array([[3,1],[2,0]])
        #subpix = np.array([[2,0],[3,1]])

        ## Enlarge if necessary
        ##----------------------
        while tmpnside < nside :
            tmpnside *= 2
            pix=np.zeros((tmpnside,tmpnside),dtype=np.int64)

            pix[0:tmpnside//2,0:tmpnside//2] = subpix + 3*(tmpnside**2)//4
            pix[tmpnside//2:tmpnside,tmpnside//2:tmpnside] = subpix
            pix[tmpnside//2:tmpnside,0:tmpnside//2] = subpix + 2*(tmpnside**2)//4
            pix[0:tmpnside//2,tmpnside//2:tmpnside] = subpix + (tmpnside**2)//4
            '''
            pix[0:tmpnside/2,0:tmpnside/2] = subpix + 2*(tmpnside**2)/4
            pix[tmpnside/2:tmpnside,tmpnside/2:tmpnside] = subpix + (tmpnside**2)/4
            pix[tmpnside/2:tmpnside,0:tmpnside/2] = subpix + 3*(tmpnside**2)/4
            pix[0:tmpnside/2,tmpnside/2:tmpnside] = subpix
            '''
            subpix = pix

        subpix += face*(nside**2)

        if ordering is 'Ring' :
            subpix = hp.nest2ring(nside, subpix)

    return subpix



## LOCAL_COV: Function to calculate local covariance of two maps
##--------------------------------------------------------------

def local_cov(map1, map2=0, mask=0, ordering='Ring', degrade=8, neigh_size=1, nocentralpix=False, silent=True, upgrade=False) :

    if len(map1.shape) > 1 :
        npix = len(map1[0])
    else :
        npix = len(map1)

    if np.isscalar(map2) :
        whok, = np.where(np.isfinite(map1)*(map1 != hp.UNSEEN))
        nwhok = len(whok)
        mapco = np.copy(map1)
        if nwhok != 0 :
            mapco[whok] *= map1[whok]
    else :
        if len(map2.shape) > 1 :
            mpix = len(map2[0])
        else :
            mpix = len(map2)
        if mpix != npix :
            print('UNMATCHED SIZES OF INPUT MAPS - QUITTING!!!')
            sys.exit()
        wh1, = np.where(np.isfinite(map1)*(map1 != hp.UNSEEN))
        wh2, = np.where(np.isfinite(map2)*(map2 != hp.UNSEEN))
        whok = list(set(wh1).intersection(wh2))
        nwhok = len(whok)
        mapco = np.tile(hp.UNSEEN,npix)
        if nwhok != 0 :
            mapco[whok] = map2[whok]*map1[whok]

    nside = hp.npix2nside(npix)

    ## First degrade a bit to speed-up smoothing
    nside_out = np.max([nside//degrade, 1])

    ## Then smooth in pixel space
    if np.isscalar(mask) :
        stat = hp.ud_grade(mapco, nside_out, pess=False, order_in=ordering, order_out='Ring', dtype=np.float64)
        if neigh_size > 0 :
            stat2 = pix_smooth(stat, ordering='Ring', maxremote=neigh_size, nocentralpix=nocentralpix, silent=silent)
            stat = np.copy(stat2)
    else :
        mapco = mapco*mask
        masq = np.copy(mask)
        wh1, = np.where((1-np.isfinite(mask))+(mask == hp.UNSEEN)+(mask == 0))
        mapco[wh1] = hp.UNSEEN
        masq[wh1] = hp.UNSEEN
        stat = hp.ud_grade(mapco, nside_out, pess=False, order_in=ordering, order_out='Ring', dtype=np.float64)
        statmask = hp.ud_grade(masq, nside_out, pess=False, order_in=ordering, order_out='Ring', dtype=np.float64)
        if neigh_size > 0 :
            stat2 = pix_smooth(stat, ordering='Ring', maxremote=neigh_size, nocentralpix=nocentralpix, silent=silent)
            stat = np.copy(stat2)
            statmask2 = pix_smooth(statmask, ordering='Ring', maxremote=neigh_size, nocentralpix=nocentralpix, silent=silent)
            statmask = np.copy(statmask2)
        whok, = np.where(np.isfinite(stat)*(stat != hp.UNSEEN)*(statmask != 0.))
        stat[whok] = stat[whok]/statmask[whok]

    ## Then upgrade back
    if upgrade :
        stat = hp.ud_grade(stat, nside, pess=False, order_in='Ring', order_out=ordering, dtype=np.float64)
    else :
        stat = hp.ud_grade(stat, nside_out, pess=False, order_in='Ring', order_out=ordering, dtype=np.float64)


    return stat



## PIX_SMOOTH : Function to smooth pixels
##---------------------------------------

def pix_smooth(map_in, ordering='Ring', maxremote=1, nocentralpix=False, silent=True) :

    n = np.max([maxremote,1])

    weights = np.tile(1.,n+1)
    for r in range(1,n+1) :
        weights[r] = 8.*r

    if nocentralpix :
        weights[0]=0.

    kerpix = np.tile(1./np.sum(weights),n+1)

    smoothmap = pix_average(map_in, maxremote=maxremote, ordering=ordering, nocentralpix=nocentralpix, silent=silent)
    #smoothmap = pix_convol(map_in, kernelpix=kerpix, ordering=ordering, silent=silent)

    return smoothmap



## PIX_AVERAGE : Function to average pixels over a neighborhood
##-------------------------------------------------------------

def pix_average(map_in, maxremote=1, ordering='Ring', nocentralpix=False, silent=True) :

    map_out = np.copy(map_in)
    nside = hp.get_nside(map_in)

    mdx, = np.where(np.isfinite(map_in)*(map_in != hp.UNSEEN))

    for ipix in mdx :

        nbrs = hpx_neighbours(ipix, maxremote, nside, ordering=ordering)
        neigs = np.copy(map_in[nbrs[0]])

        if nocentralpix :
            neigs = neigs[1::]

        ndx, = np.where(np.isfinite(neigs)*(neigs != hp.UNSEEN))

        map_out[ipix] = np.mean(neigs[ndx])


    return map_out




## PIX_CONVOL : Function to convolve pixels with a kernel
##-------------------------------------------------------

def pix_convol(map_in, kernelpix=np.array([1,1]), ordering='Ring', silent=True) :

    ## Create output array
    map_out = np.copy(map_in)
    nside = hp.get_nside(map_in)

    ## Make 2-D kernel
    nhalf = kernelpix.size
    max_remote = nhalf-1
    nsidekern = 2*nhalf-1
    kernel2d = np.zeros((nsidekern,nsidekern), dtype=np.float64)
    kernel2d[nhalf-1,nhalf-1] = kernelpix[0]
    if nhalf >= 2 :
        for i in range(1,nhalf) :
            kernel2d[nhalf-i-1,nhalf-i-1:nhalf+i] = kernelpix[i]
            kernel2d[nhalf+i-1,nhalf-i-1:nhalf+i] = kernelpix[i]
            kernel2d[nhalf-i-1:nhalf+i,nhalf-i-1] = kernelpix[i]
            kernel2d[nhalf-i-1:nhalf+i,nhalf+i-1] = kernelpix[i]

    ## Smooth all faces
    for face in range(12) :
        if silent is False :
            print('Convolution for face '+ str(face))

        ## Find pixels for face
        pixface = face_pix(nside, face=face, ordering=ordering)

        ## Read map_in values
        face_in = np.array(map_in[pixface])

        ## Smooth the face
        if kernel2d.size <= face_in.size :
            face_in = signal.convolve2d(face_in, kernel2d, 'same', 'fill')

        ## Assign values to output map
        map_out[pixface] = face_in

        ## Take care of the borders, which are wrong for the moment
        if nhalf > 1 :
            ## Find border pixels, not smoothed in an appropriate way

            borderpix1 = pixface[0:(nside-nhalf+1),0:(nhalf-1)]
            borderpix2 = pixface[(nside-nhalf+1):nside,0:(nside-nhalf+1)]
            borderpix3 = pixface[(nhalf-1):nside,(nside-nhalf+1):nside]
            borderpix4 = pixface[0:(nhalf-1),(nhalf-1):nside]

            borderpix = np.concatenate((borderpix1.reshape(borderpix1.size),borderpix2.reshape(borderpix2.size),
                                        borderpix3.reshape(borderpix3.size), borderpix4.reshape(borderpix4.size)))
            nb = borderpix.size

            ## For computing factor for normalisation, to take into account the number
            ## of pixels in each ring of 'remoteness'
            factor = np.tile(1.,nhalf)
            expected = np.tile(1.,nhalf)
            for r in range(1,nhalf) :
                expected[r] = 8.*r

            ## Loop on border pixels, using hpx_pix_neighbours
            for p in range(nb) :
                ipix = borderpix[p]
                nbrs = hpx_neighbours(ipix, max_remote, nside, ordering=ordering)
                rmote = np.copy(nbrs[1])
                neigs = np.copy(nbrs[0])
                wh, = np.where(rmote == max_remote)
                nwh = wh.size
                for x in range(1,max_remote+1) :
                    wh, = np.where(rmote == x)
                    nwh = wh.size
                    if nwh == 0:
                        print('x = '+str(x)+', remote = ', rmote)
                        print('ipix = '+str(ipix)+', neighbor size = ', max_remote)
                        print('ordering =', ordering)
                        print('nside =', nside)
                        sys.exit()
                    factor[x] = expected[x]/nwh
                f = factor*kernelpix
                map_out[ipix] = np.sum(map_in[neigs]*f[rmote])

    return map_out



### ROTATE_MAP : Procedure to change the coordinate system of a polarization map
###-----------------------------------------------------------------------------
def rotate_map(map_in, coord_in='G', coord_out='C') :

    map_out = np.copy(map_in)

    nside = hp.get_nside(map_in)
    npix = hp.nside2npix(nside)

    idx = np.arange(npix)
    ang = hp.pix2ang(nside, idx)

    ## Find angular positions in old coords
    rinv = hp.Rotator(coord=[coord_out,coord_in])

    ang_rot = rinv(ang[0], ang[1])

    if np.isscalar(map_in[0]) :

        ## Assign values from old coords
        map_out = hp.get_interp_val(map_in, ang_rot[0], ang_rot[1])

    else :

        for k in range(len(map_in)) :
            ## Assign values from old coords
            map_out[k] = hp.get_interp_val(map_in[k], ang_rot[0], ang_rot[1])

        Qmap = np.copy(map_out[-2])
        Umap = np.copy(map_out[-1])

        ## Transform old i.j,k unit vectors in new coords (r-hat)
        r = hp.Rotator(coord=[coord_in,coord_out])

        x_ang = np.array([r(np.pi/2,0), r(np.pi/2,np.pi/2), r(0,0)])
        r_hat = np.zeros((3,3))

        for k in range(3) :
            r_hat[k] = np.array([np.sin(x_ang[k,0])*np.cos(x_ang[k,1]), np.sin(x_ang[k,0])*np.sin(x_ang[k,1]), np.cos(x_ang[k,0])])

        ## Healpix polarization convention: e1 = theta-hat, e2 = phi-hat
        ## Write new theta-hat, phi-hat unit vectors in new x,y,z coords
        th_v = np.array([np.cos(ang[0])*np.cos(ang[1]), np.cos(ang[0])*np.sin(ang[1]), -np.sin(ang[0])])
        ph_v = np.array([-np.sin(ang[1]), np.cos(ang[1]), np.zeros(npix)])

        th_r = np.copy(th_v)

        for j in range(3) :
            ## Write old theta-hat, phi-hat unit vectors in new x,y,z coords
            th_r[j]  = np.cos(ang_rot[0])*np.cos(ang_rot[1])*r_hat[0,j]
            th_r[j] += np.cos(ang_rot[0])*np.sin(ang_rot[1])*r_hat[1,j] - np.sin(ang_rot[0])*r_hat[2,j]

        ## Find the rotation angle between old and new e1, e2
        cos = np.sum(th_r*th_v, axis=0)
        sin = -np.sum(th_r*ph_v, axis=0)

        alph = np.arctan2(sin,cos)

        cos2a = np.cos(2*alph)
        sin2a = np.sin(2*alph)

        map_out[1] = cos2a*Qmap + sin2a*Umap
        map_out[2] = cos2a*Umap - sin2a*Qmap

    return map_out



### CHANGE_RESOLUTION : Procedure to change the resolution of a map
###----------------------------------------------------------------
def change_resolution(map_in, map_out=False, beam_in=0, beam_out=0, nside_out=None, l_max=None):

    nside_in = hp.get_nside(map_in)

    if not nside_out :
        nside_out = nside_in

    if not l_max :
        l_max = 2*nside_out
        if l_max > 2*nside_in :
            l_max = 2*nside_in

    npol = 1
    pols = False
    if not np.isscalar(map_in[0]) :
        pols = True
        npol = len(map_in)

    if pols :
        alms = []
        for ip in range(npol):
            alms += [hp.map2alm(map_in[ip], lmax=l_max)]
        alms = np.array(alms)
    else :
        alms = np.array(hp.map2alm(map_in, lmax=l_max))

    if np.isscalar(beam_in) :
        bl_in = hp.gauss_beam(beam_in*np.pi/180./60., lmax=l_max, pol=pols)
        bl_in = bl_in.T
    else :
        bl_in = beam_in

    if (len(bl_in) < npol) and (not np.isscalar(bl_in[0])) :
        for ip in range(len(bl_in),npol) :
            bl_in = np.concatenate((bl_in,[bl_in[-1]]))

    if pols :
        for ip in range(npol):
            alms[ip] = hp.almxfl(alms[ip], 1./bl_in[ip])
    else :
        alms = hp.almxfl(alms, 1./bl_in)

    if np.isscalar(beam_out) :
        bl_ou = hp.gauss_beam(beam_out*np.pi/180./60., lmax=l_max, pol=pols)
        bl_ou = bl_ou.T
    else :
        bl_ou = beam_out

    if (len(bl_ou) < npol) and (not np.isscalar(bl_ou[0])) :
        for ip in range(len(bl_ou),npol) :
            bl_ou = np.concatenate((bl_ou,[bl_ou[-1]]))

    if pols :
        for ip in range(npol):
            alms[ip] = hp.almxfl(alms[ip], bl_ou[ip])
    else :
        alms = hp.almxfl(alms, bl_ou)

    if map_out :
        if pols :
            mout = []
            for ip in range(npol):
                mout += [hp.alm2map(alms[ip], nside_out, verbose=False)]
            mout = np.array(mout)
        else :
            mout = hp.alm2map(alms, nside_out, verbose=False)
    else :
        mout = alms

    return mout



### SMOOTH_COVARIANCE : Procedure to smooth the noise covariance map
###-----------------------------------------------------------------
def smooth_covariance(map_in, beam_in=0, beam_out=0, nside_out=None, l_max=None):

    nside_in = hp.get_nside(map_in)
    if not nside_out :
        nside_out = nside_in

    if not l_max :
        l_max = 2*nside_out

    fac = (1.*nside_in/nside_out)**2

    polar = True
    if np.isscalar(map_in[0]) :
        polar = False

    wl_in = np.array(hp.pixwin(nside_in, pol=polar))

    if np.isscalar(beam_in) :
        bl_in = hp.gauss_beam(beam_in*np.pi/180./60., lmax=l_max)
    else :
        bl_in = beam_in

    if polar :
        wl_in[1,:2] = 1
        bt_in = (bl_in[:l_max+1]**2)*wl_in[0,:l_max+1]
        bp_in = (bl_in[:l_max+1]**2)*wl_in[1,:l_max+1]
        btp_in = (bl_in[:l_max+1]**2) * np.sqrt(wl_in[0,:l_max+1]*wl_in[1,:l_max+1])
    else :
        bl_in = (bl_in[:l_max+1]**2)*wl_in[:l_max+1]

    wl_out = np.array(hp.pixwin(nside_out, pol=polar))

    if np.isscalar(beam_out) :
        bl_out = hp.gauss_beam(beam_out*np.pi/180./60., lmax=l_max)
    else :
        bl_out = beam_out

    if polar :
        wl_out[1,:2] = 1
        bt_out = (bl_out[:l_max+1]**2)*wl_out[0,:l_max+1]
        bp_out = (bl_out[:l_max+1]**2)*wl_out[1,:l_max+1]
        btp_out = (bl_out[:l_max+1]**2) * np.sqrt(wl_out[0,:l_max+1]*wl_out[1,:l_max+1])
    else :
        bl_out = (bl_out[:l_max+1]**2)*wl_out[:l_max+1]

    if np.isscalar(map_in[0]) :
        map_out = change_resolution(map_in, map_out=True, beam_in=bl_in, beam_out=bl_out, nside_out=nside_out, l_max=l_max)
    else :
        npol = len(map_in)
        map_out = np.zeros((npol,hp.nside2npix(nside_out)))
        for k in range(npol) :
            if k == 0 :
                bl_in = bt_in
                bl_out = bt_out
            elif k < 3 :
                bl_in = btp_in
                bl_out = btp_out
            else :
                bl_in = bp_in
                bl_out = bp_out

            wh0, = np.where(bl_out < 1e-12)
            if len(wh0) :
                bl_out[wh0] = 1e-12

            map_out[k] = change_resolution(map_in[k], map_out=True, beam_in=bl_in, beam_out=bl_out, nside_out=nside_out, l_max=l_max)

    return map_out/fac


### MAKE_NOISE : Procedure to make noise realization given a covariance map
###------------------------------------------------------------------------
def make_noise(cov_in) :

    if np.isscalar(cov_in[0]) :
        whunseen, = np.where((cov_in <= 0)+(1-np.isfinite(cov_in)))
        if len(whunseen) :
            cov_in[whunseen] = 1e-12
        mean = np.zeros(len(cov_in))
        noise = np.random.normal(mean, np.sqrt(cov_in))
    else :
        for k in range(len(cov_in)) :
            whunseen, = np.where((cov_in[k] < -1e10)+(1-np.isfinite(cov_in[k])))
            if len(whunseen) :
                cov_in[k,whunseen] = 1e-12
        mean = np.zeros(3)
        noise = np.copy(cov_in[:3])
        for ip in range(len(cov_in[1])) :
            Cmat = np.array([[cov_in[0,ip], cov_in[1,ip], cov_in[2,ip]],
                             [cov_in[1,ip], cov_in[3,ip], cov_in[4,ip]],
                             [cov_in[2,ip], cov_in[4,ip], cov_in[5,ip]]])
            noise[:,ip] = np.random.multivariate_normal(mean, Cmat)

    return noise



### LOWER_POWERof2 : Function to Calculate Lower Power of 2
###--------------------------------------------------------

def lower_powerof2(number) :

    if number <= 0 :
        print('No power of 2 lower than number')
    elif number < 1 :
        return 1./lower_powerof2(1./number)/2
    else :
     n = number
     cont = 1
     val = 1
     while cont == 1 :
        val = 2*val
        if val > n :
            cont = 0
     return int(val//2)



### GET_RESOLUTION : Function for determining angular scale corresponding to a given multipole scale
###-------------------------------------------------------------------------------------------------

def get_resolution(lmax, b_frac=0.1, arcmin=True) :

    b_frac = min([b_frac,0.1])

    si = np.sqrt(-np.log(b_frac)*2/lmax/(lmax+1))

    if arcmin :
        b_res = si*180.*60.*np.sqrt(8.*np.log(2.)) / np.pi
    else :
        b_res = si*180.*np.sqrt(8.*np.log(2.)) / np.pi

    return b_res



### GAUSS_BANDS : Function for gaussian windows in multipole space
###---------------------------------------------------------------

def gauss_bands(bnd_res, lmax, ana_bandfile=None, syn_bandfile=None, verbose=True) :

    ## Floating point numpy array needlet resoliutions :
    ##      bnd_res = np.array([300.,120.,60.,45.,30.,15.,10.,7.5,5.])
    bnd_res = np.flip(np.sort(np.array(bnd_res)))

    sj = np.array(bnd_res) * np.pi / 180. / 60. / np.sqrt(8.*np.log(2.))

    ell = np.arange(lmax+1)

    blj = np.exp(-np.outer(sj**2,ell*(ell+1.)/2.))

    bands = np.zeros((blj.shape[0]+1,blj.shape[-1]))

    for zk in range(len(blj)-1) :
        bands[zk+1] = np.sqrt(blj[zk+1]-blj[zk])

    bands[0] = np.sqrt(blj[0])
    bands[-1] = np.sqrt(1-blj[-1])

    bandcenters = np.zeros(len(bands))
    for kz in range(len(bands)) :
        wh, = np.where(bands[kz] == max(bands[kz]))
        bandcenters[kz] = wh[0]
        if syn_bandfile :
            bands[kz] *= max(bands[kz])

    if verbose :
        print('\nGaussian Band Centers = '+str(bandcenters))

    if syn_bandfile :
        np.save(syn_bandfile, bands)
        for kz in range(len(bands)) :
            bands[kz] /= max(bands[kz])

    if ana_bandfile :
        np.save(ana_bandfile, bands)

    return bands, bandcenters.astype(int)


### COSINE_BANDS : Function for cosine windows in multipole space
###--------------------------------------------------------------

def cosine_bands(bandcenters=0, bndfile=None) :

    if np.isscalar(bandcenters) :
        bandcenters = np.array([0,20,40,70,100,140,180,240,320,450,600,800,1000,1300,1600,2000,2500,3000,4000,6000])

    lmax = np.max(bandcenters)
    nbands = len(bandcenters)
    bands = np.zeros((nbands, lmax+1), dtype=np.float64)

    b=bandcenters[0]
    c=bandcenters[1]

    ell_left = np.arange(b+1)
    if b > 0 :
        bands[0, ell_left] = 1.

    ell_right = b + np.arange(c-b+1)
    bands[0,ell_right] = np.cos(np.pi*(ell_right-b)/(c-b)/2.)

    if nbands >= 3 :
        for i in range(1,nbands-1) :
            a=bandcenters[i-1]
            b=bandcenters[i]
            c=bandcenters[i+1]

            ## Left part
            ell_left = a + np.arange(b-a+1)
            bands[i,ell_left] = np.cos(np.pi*(b-ell_left)/(b-a)/2.)
            ## Right part
            ell_right = b + np.arange(c-b+1)
            bands[i,ell_right] = np.cos(np.pi*(ell_right-b)/(c-b)/2.)

        a=bandcenters[nbands-2]
        b=bandcenters[nbands-1]
        ## Left part
        ell_left = a + np.arange(b-a+1)
        bands[nbands-1,ell_left] = np.cos(np.pi*(b-ell_left)/(b-a)/2.)

    if bndfile :
        np.save(bndfile, bands)

    return bands


### ALM2NEEDLETS : Function to get needlets from alms
###--------------------------------------------------

def alm2needlets(alm, needletsfile, bands, degrade=False, needletnside=None, nside_max_w=None, nside_min_w=None) :

    needout = []
    almlmax = hp.Alm.getlmax(len(alm))

    uselmax = np.max(np.where(bands[0] != 0))
    nbands = len(bands)
    if nbands > 1 :
        for i in range(1, nbands) :
            uselmax = np.max([np.max(np.where(bands[i] != 0)) , uselmax])

    if needletnside :
        nside = needletnside
    else :
        nside = lower_powerof2(uselmax)
    if nside_min_w :
        nside = np.max([nside , nside_min_w])
    if nside_max_w :
        nside = np.min([nside , nside_max_w])

    lmax = len(bands[0])-1
    if lmax > almlmax :
        print('WARNING: Bands extend beyond the maximum ell for alm!!!')

    nbands = len(bands)

    for i in range(nbands) :

        uselmax = np.max(np.where(bands[i] != 0)) +1

        if degrade :
            possiblenside = np.array([1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192])
            whereok, = np.where(possiblenside > uselmax/2)
            usenside = np.min(possiblenside[whereok])
            if nside_min_w :
                usenside = np.max([usenside , nside_min_w])
            if nside_max_w :
                usenside = np.min([usenside , nside_max_w])
        else :
            usenside = nside
            if nside_min_w :
                usenside = np.max([usenside , nside_min_w])
            if nside_max_w :
                usenside = np.min([usenside , nside_max_w])

        if needletnside :
            usenside = needletnside

        sonu = hp.Alm.getsize(lmax)
        alm_write = np.zeros(sonu, dtype=np.complex128)

        if sonu > len(alm):
            sonu = len(alm)

        alm_write[0:sonu] = alm[0:sonu]
        alm_write[sonu-1::] = alm[sonu-1]

        alm_write = hp.almxfl(alm_write, bands[i])
        needlets_write = hp.alm2map(alm_write, usenside, verbose=False)

        outfile = needletsfile + '_scale' + str(i).zfill(2) + '.fits'
        hp.write_map(outfile, needlets_write, overwrite=True)
        needout.append(outfile)

    return needout


### NEEDLETS2ALM : Function to get alms from needlets
###--------------------------------------------------

def needlets2alm(needlets, bands, nolastwindow=0, scale_map=None, verbose=False) :

    nwindows = len(bands)

    nw=nwindows-nolastwindow
    windows = bands[0:nw,:]

    win_lmax = len(windows[0]) - 1
    nalm = hp.Alm.getsize(win_lmax)
    alm = np.zeros(nalm, dtype=np.complex128)

    for i in range(nw) :
        maps = hp.read_map(needlets[i], verbose=verbose)
        almtemp = hp.map2alm(maps, lmax=win_lmax)
        almtemp = hp.almxfl(almtemp, windows[i])
        alm += almtemp
        if scale_map :
            smap = hp.alm2map(almtemp, 512, verbose=False)
            hp.write_map(scale_map+'_scale'+str(i).zfill(2)+'.fits', smap, overwrite=True)

    return alm


### Function to Calculate Number of Degrees of Freedom of Foreground Signal
###------------------------------------------------------------------------
def num_dof(mu, en, use_mdl=False) :

    d = len(mu)
    aic = np.zeros(d)
    for k in range(d) :
        lg = 0
        gs = 0
        for i in range(k, d) :
            x = mu[i]
            gs += mu[i]
            if x == 0. :
                x = 1.e-12
            x = np.abs(x)
            lg += np.log(x)

        if gs == 0 :
            gs = 1.e-12

        carp = k*(2*d-k)
        if use_mdl :
            ##carp += 1.
            carp *= np.log(en)/2.

        aic[k] = en*(d-k)*np.log(gs/(d-k)) - en*lg + carp

    wh, = np.where(aic == aic.min())

    return wh[0]


### Function to Calculate Number of Degrees of Freedom of Foreground Signal
### Applying Akaike Information Criterion
###------------------------------------------------------------------------
def akaike(mu) :

    d = len(mu)
    aic = np.zeros(d)
    for k in range(d) :
        gs = 0
        for i in range(k, d) :
            gs += mu[i]
            if mu[i] == 0. :
                mu[i] = 1.e-12
            mu[i] = np.abs(mu[i])
            gs -= np.log(mu[i])
            gs -= 1

        aic[k] = 2*(k+1) + gs

    wh, = np.where(aic == aic.min())

    return wh[0]


## COS_TRANSITION_MASK: Function to get a latitude mask with a cos squared apodisation
##------------------------------------------------------------------------------------

def cos_transition_mask (rangelon, trans, nside=2048, Nest=False) :


    npix = hp.nside2npix(nside)
    indx = np.arange(npix, dtype=np.int64)

    angs = hp.pix2ang(nside, indx, nest=Nest)
    indx = 0
    phi = 0
    lat = abs(90. - angs[0]*180./np.pi)

    phi=0
    lat = abs(lat)

    wh, = np.where(lat > (rangelon+trans))
    mask = np.zeros(len(lat))
    mask[wh] = 1.

    if trans > 0. :
        wh, = np.where((lat >= rangelon) & (lat <= (rangelon+trans)))
        select = np.zeros(len(lat))
        select[wh] = 1.
        mask += select*(np.sin(np.pi*(lat-rangelon)/trans/2.)**2)

    return mask



## LUDWIG2QU: Function to get the Q and U maps of unit polarisation in the Ludwig3 convention of co-polarisation
##--------------------------------------------------------------------------------------------------------------

def ludwig2qu (inmap, pole='north', Nest=False) :

    nside = hp.get_nside(inmap)
    ipix = np.arange(hp.nside2npix(nside))
    theta_phi = hp.pix2ang(nside, ipix, nest=Nest)
    twophi = 2*theta_phi[1]

    Q = np.cos(twophi)
    U = np.sin(twophi)

    if pole == 'south' :
        U = -U

    tmap = np.copy(inmap)

    tqu1 = np.array([tmap, Q*tmap, U*tmap])
    tqu2 = np.array([tmap, U*tmap, Q*tmap])

    return (tqu1, tqu2)


## APODIZE_MASK: procedure to apodize a 0-1 mask, for a smooth transition between the two regions
##-----------------------------------------------------------------------------------------------

def apodize_mask (mask, length, deg=True) :

    dist = distance_to_mask(mask, maxdist=length, deg=deg)

    apmask = np.sin((np.pi/2.)*(dist/length))

    return (apmask)**2


## DISTANCE_TO_MASK: procedure to find distance to a mask, up to some max (for apodisation purposes)
##--------------------------------------------------------------------------------------------------

def distance_to_mask (mask, maxdist=None, deg=True) :

    nside = hp.npix2nside(len(mask))

    if maxdist == None :
        if deg :
            maxdist = 180.
        else :
            maxdist = np.pi

    p0, p1 = border_of_mask(mask)

    dist = maxdist * mask

    if len(p0) :
        for p in range(len(p0)) :
            vec = np.array(hp.pix2vec(nside, p0[p]))
            listpix, d = query_dist(nside, vec, maxdist, deg=deg)
            if len(listpix) :
                tmpdist = dist[listpix]
                wh, = np.where(d < tmpdist)
                tmpdist[wh] = d[wh]
                dist[listpix] = tmpdist

    return dist * mask


## BORDER_OF_MASK: function to find border pixels in a mask
##---------------------------------------------------------

def border_of_mask (mask) :

    npix = len(mask)
    nside = hp.npix2nside(npix)

    wh, = np.where(mask == 1)

    if len(wh) == 0 :
        p0 = np.array([])
        p1 = np.array([])

        return (p0, p1)

    pixsize = np.sqrt(4.*np.pi/npix) * 180.*60./np.pi

    bm_o = pixsize * np.sqrt(2.)

    smoothmask = change_resolution(mask, map_out=True, beam_out=bm_o)

    wh1, = np.where(mask == 0)
    wh2, = np.where(smoothmask[wh1] >= 0.15)
    p0 = wh1[wh2]

    wh1, = np.where(mask == 1)
    wh2, = np.where(smoothmask[wh1] <= 0.85)
    p1 = wh1[wh2]

    if len(p0) :

        m = 0*mask
        m[p0] = 1
        m[p1] = 1

        for p in range(len(p0)) :
            ip, rmt = hpx_neighbours(p0[p], 1, nside)
            m[ip] = 1

        m = m*mask
        p1, = np.where(m == 1)

    if len(p1) :

        m = 0*mask
        m[p0] = 1
        m[p1] = 1

        for p in range(len(p1)) :
            ip, rmt = hpx_neighbours(p1[p], 1, nside)
            m[ip] = 1

        m = m*(1-mask)
        p0, = np.where(m == 1)

    return (p0, p1)


## QUERY_DIST: procedure to get pixels within certain distance of a point, and their distance
##-------------------------------------------------------------------------------------------

def query_dist(nside, vector0, radius_in, deg=True) :

    if deg :
        rad = radius_in*np.pi/180.

    listpix = hp.query_disc(nside, vector0, rad)

    if len(listpix) :
        vec_out = np.array(hp.pix2vec(nside, listpix))

        cosang = vector0.dot(vec_out)

        wh, = np.where(cosang > 1)
        cosang[wh] = 1.

        distpix = np.arccos(cosang)

        if deg :
            distpix = distpix * 180./np.pi

    else :
        distpix = 0.

    return (listpix, distpix)



## MATRIX_SOn: procedure to build rotation matrix R in SO(n) given a vector "u" such that R[0] = u :
##-------------------------------------------------------------------------------------------------
def matrix_SOn(u) :

    a = u/np.sqrt(sum(u**2))

    dim = len(a)
    rot1 = np.zeros(dim)
    rot2 = np.zeros(dim)

    Rrot = np.eye(dim)

    # Unit x-vector bir = (1,0,..) :
    bir = np.zeros(dim)
    bir[0] = 1.

    if dim == 2 :
        rot1[0] = np.max(np.sqrt(1.-a**2))
        rot1[1] = np.sqrt(2.-np.sum(a)**2) - rot1[0]
        det1 = a[0]*rot1[1] - a[1]*rot1[0]
        if det1 < 0 :
            rot1 = -rot1
            Rrot = np.array([a,rot1])
    elif np.sum((a - bir)**2) < 1e-6 :
        Rrot = np.eye(dim)
    else :
        # Unit vector r1 = a - bir :
        rot1 = a - bir
        rot1 /= np.sqrt(np.dot(rot1,rot1))

        # Reflection about orthogonal plane of r1, R1: a <-> bir :
        Rrot1 = np.eye(dim) - 2.*np.outer(rot1,rot1)

        # Unit vector r2 perpendicular to both a and b
        whb0, = np.where(a[1:] == 0)
        if len(whb0) :
            rot2[whb0[0]+1] = 1.
        else :
            rot2[1] = a[2]
            rot2[2] = -a[1]

        rot2 /= np.sqrt(np.dot(rot2,rot2))

        # Reflection about orthogonal plane of r2, R2: a -> a, b -> b :
        Rrot2 = np.eye(dim) - 2.*np.outer(rot2,rot2)

        # Two reflections give a rotation around the intermediate axis, R: a <-> b
        # Since b = (1,0,0,..) => R[0] = a :
        Rrot = np.dot(Rrot1,Rrot2)

    return Rrot

## proj_MorthV: project matrix M onto subspace orthogonal to vector V :
##-------------------------------------------------------------------------------------------------
def proj_MorthV(M, v) :

    a = np.copy(v)/np.sqrt(v.dot(v))
    t = np.copy(M.T)

    for c in range(len(M.T)) :
        t[c] = M.T[c] - (M.T[c].dot(a))*a

    return t.T

## STR2BOOL: function to convert string into boolean
##---------------------------------------------------------

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


### Planck blackbody
###------------------------------------------------------------------------------
def planck_bb(nu, temp) :

    c_vac = 299792458.
    h_planck = 6.62607554e-34
    k_boltzman = 1.38065812e-23
    T_CMB = 2.725480

    nu1 = nu*1e9

    return 2*(h_planck*nu1)*(nu1/c_vac)**2 / (np.exp(h_planck*nu1/k_boltzman/temp) - 1.)


### Derivative of Planck blackbody dB(nu)/dT
###------------------------------------------------------------------------------
def planck_dBnu_dT(nu,temp) :

    c_vac = 299792458.
    h_planck = 6.62607554e-34
    k_boltzman = 1.38065812e-23
    T_CMB = 2.725480

    nu1 = nu*1e9

    b_nu = planck_bb(nu, temp)

    return (b_nu*c_vac/nu1/temp)**2 / 2. * np.exp(h_planck*nu1/k_boltzman/temp) / k_boltzman


### Procedure to calculate the conversion factor from K_RJ to K_CMB
###------------------------------------------------------------------------------
def KRJ2KCMB(nu):

    c_vac = 299792458.
    h_planck = 6.62607554e-34
    k_boltzman = 1.38065812e-23
    T_CMB = 2.725480

    nu1 = nu*1e9

    fac_1 = planck_dBnu_dT(nu,T_CMB)
    fac_2 = 2*k_boltzman*(nu1/c_vac)**2

    return fac_2/fac_1


