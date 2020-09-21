import numpy as np

def calcMueller(images, radians_light, radians_camera):
    """
    Calculate mueller matrix from captured images and 
    angles of the linear polarizer on the light side and the camera side.
    
    Parameters
    ----------
    images : np.ndarray, (height, width, N)
        Captured images
    radians_light : np.ndarray, (N,)
        polarizer angles on the light side
    radians_camera : np.ndarray, (N,)
        polarizer angles on the camera side
    Returns
    -------
    img__mueller : np.ndarray, (height, width, 9)
        Calculated mueller matrix image
    """
    cos_light  = np.cos(2*radians_light)
    sin_light  = np.sin(2*radians_light)
    cos_camera = np.cos(2*radians_camera)
    sin_camera = np.sin(2*radians_camera)
    A = np.array([np.ones_like(radians_light), cos_light, sin_light, cos_camera, cos_camera*cos_light, cos_camera*sin_light, sin_camera, sin_camera*cos_light, sin_camera*sin_light]).T
    A_pinv = np.linalg.inv(A.T @ A) @ A.T #(9, depth)
    img_mueller = np.tensordot(A_pinv, images, axes=(1,2)).transpose(1, 2, 0) #(height, width, 9)
    #height, width, _ = images.shape
    #img_mueller_2D = np.reshape(img_mueller, (height, width, 3, 3))
    return img_mueller


def plotMueller(filename, img_mueller, vabsmax=None, dpi=300, cmap="RdBu", add_title=True):
    """
    Apply color map to the Mueller matrix image and save them side by side
    
    Parameters
    ----------
    filename : str
        File name to be written.
    img_mueller : np.ndarray, (height, width, 9) or (height, width, 16)
        Mueller matrix image.
    vabsmax : float
        Absolute maximum value for plot. If None, the absolute maximum value of 'img_mueller' will be applied.
    dpi : float
        The resolution in dots per inch.
    cmap : str
        Color map for plot.
        https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    add_title : bool
        Whether to insert a title (e.g. m11, m12...) in the image.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    
    # Check for 'img_muller' shape
    height, width, channel = img_mueller.shape
    if channel==9:
        n = 3
    elif channel==16:
        n = 4
    else:
        raise ValueError(f"'img_mueller' shape should be (height, width, 9) or (height, width, 16): ({height}, {width}, {channel})")
    
    def add_inner_title(ax, title, loc, size=None, **kwargs):
        """
        Insert the title inside image
        """
        from matplotlib.offsetbox import AnchoredText
        from matplotlib.patheffects import withStroke
        if size is None:
            size = dict(size=plt.rcParams['legend.fontsize'])
        at = AnchoredText(title, loc=loc, prop=size,
                          pad=0., borderpad=0.5,
                          frameon=False, **kwargs)
        ax.add_artist(at)
        at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
        return at
        
    # Vreta figure
    fig = plt.figure()

    # Create image grid
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(n,n),
                     axes_pad=0.0,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="3%",
                     cbar_pad=0.10,
                     )
    
    # Set absolute maximum value
    vabsmax = np.max(np.abs(img_mueller)) if (vabsmax is None) else vabsmax
    vmax =  vabsmax
    vmin = -vabsmax

    # Add data to image grid
    for i, ax in enumerate(grid):
        # Remove the ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add title
        if add_title:
            maintitle = "$m$$_{0}$$_{1}$".format(i//n+1, i%n+1) # m{}{}
            t = add_inner_title(ax, maintitle, loc='lower right')
        
        # Add image
        im = ax.imshow(img_mueller[:,:,i],
                       vmin=vmin,
                       vmax=vmax,
                       cmap=cmap,
                       )

    # Colorbar
    cbar = ax.cax.colorbar(im, ticks=[vmin, 0, vmax])
    cbar.solids.set_edgecolor("face")
    ax.cax.toggle_label(True)
    
    # Save figure
    plt.savefig(filename, bbox_inches='tight', dpi=dpi)
