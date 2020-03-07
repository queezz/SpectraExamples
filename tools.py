import numpy as np
import os

def latex_float(f,prec=2):
    float_str = "{0:.{1}e}".format(f,prec)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

def html_float(f,prec=2):
    float_str = "{0:.{1}e}".format(f,prec)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return u"{0}x10<sup>{1}</sup>".format(base, int(exponent))
    else:
        return float_str

def savitzky_golay(y, window_size, order):
    ''' smoothing '''
    from math import factorial
    from scipy.signal import savgol_filter
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        window_size += 1
    return savgol_filter(y,window_size,order)

def fft_tool(x,y):
    ''' returns power spectrum for the given x, y signal '''
    dx = np.diff(x)
    uniform = not np.any(np.abs(dx-dx[0]) > (abs(dx[0]) / 1000.))
    if not uniform:
        import scipy.interpolate as interp
        x2 = np.linspace(x[0], x[-1], len(x))
        y = interp.griddata(x, y, x2, method='linear')
        x = x2
    f = np.fft.fft(y) / len(y)
    y = abs(f[1:len(f)/2])
    dt = x[-1] - x[0]
    x = np.linspace(0, 0.5*len(x)/dt, len(y))
    return x, y

def integrate(x,y):
    ''' returns integral(x) for given (x,y) array'''
    intgrated = []
    for i,_ in enumerate(y):
        yy = y[0:i+1]
        xx = x[0:i+1]
        current_int = np.trapz(yy, xx)
        intgrated.append(current_int)
    return np.array(x),np.array(intgrated)

def ind_1st_appear(arr):
    'print indexes of first appearance of the consecuent dublicates in a list'
    l = len(arr)
    i = 0
    idd = []
    ss = arr[::-1]
    while i < l-1:
        #print ss[i+1]
        if ss[i+1] != ss[i]:
            #print ss[i], ss[i] == arr[l-1-i], i, l-1-i
            idd.append(l-1-i)
        i+=1
    if ss[-1] != ss[-2]:
        idd.append(0)
    return idd[::-1]

# Matploblib adjustments
def ticks_visual(ax,**kwarg):
    '''
    makes auto minor and major ticks for matplotlib figure
    makes minor and major ticks thicker and longer
    '''
    which = kwarg.get('which','both')
    from matplotlib.ticker import AutoMinorLocator
    if which == 'both' or which == 'x':
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if which == 'both' or which == 'y':
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    l1 = kwarg.get('l1',7)
    l2 = kwarg.get('l2',4)
    w1 = kwarg.get('w1',1.)
    w2 = kwarg.get('w2',.8)
    ax.xaxis.set_tick_params(width= w1,length = l1,which = 'major')
    ax.xaxis.set_tick_params(width= w2,length = l2,which = 'minor')
    ax.yaxis.set_tick_params(width= w1,length = l1,which = 'major')
    ax.yaxis.set_tick_params(width= w2,length = l2,which = 'minor')
    return

def grid_visual(ax, alpha = [.1,.3]):
    '''
    Sets grid on and adjusts the grid style.
    '''
    ax.grid(which = 'minor',linestyle='-', alpha = alpha[0])
    ax.grid(which = 'major',linestyle='-', alpha = alpha[1])
    return

def gritix(**kws):
    '''
    Automatically apply ticks_visual and grid_visual to the
    currently active pylab axes.
    '''
    import matplotlib.pylab as plt
    
    ticks_visual(plt.gca())
    grid_visual(plt.gca())
    return

def plot_as_emf(figure, **kwargs):
    """
    Save matplotlib figure as svg,
    Convert it using inkscape to *.emf file
    
    inkscape = path to inkscape.exe
    inkscape_path = kwargs.get('inkscape', "C:\\Program Files\\Inkscape\\inkscape.exe")
    """
    import subprocess, os

    inkscape_path = kwargs.get('inkscape', "C:\\Program Files\\Inkscape\\inkscape.exe")
    filepath = kwargs.get('filename', None)

    if filepath is not None:
        path, filename = os.path.split(filepath)
        filename, extension = os.path.splitext(filename)

        svg_filepath = os.path.join(path, filename+'.svg')
        emf_filepath = os.path.join(path, filename+'.emf')

        figure.savefig(svg_filepath, format='svg')

        subprocess.call([inkscape_path, svg_filepath, '--export-emf', emf_filepath])
        os.remove(svg_filepath)


def plot_3D(x, y, p, v_init = [30,45],cmap = 'viridis'):
    '''Creates 3D plot with appropriate limits and viewing angle

    Parameters:
    ----------
    x: array of float
        nodal coordinates in x
    y: array of float
        nodal coordinates in y
    p: 2D array of float
        calculated potential field
    '''
    import matplotlib.pylab as plt
    
    fig = plt.figure(figsize=(11,7), dpi=100)
    ax = fig.gca(projection='3d')
    X,Y = np.meshgrid(x,y)
    surf = ax.plot_surface(X,Y,p[:], rstride=1, cstride=1, cmap=cmap,
            linewidth=0.1, antialiased=False)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.view_init(v_init[0],v_init[1])

def lam(p,gas='Air'):
    '''given pressure in [Pa] returns mean free path in [m]'''
    import scipy.constants as sc
    k = dict()
    #in 1e-3 cm*Torr
    k.update([('Xe',3),('H2O',3.4),('CO2',3.34),
              ('Cl',3.47),('Kr',4.05),('Ar',5.07),
              ('N2',5.1),('Air',5.1),('O2',5.4),
              ('Hg',6.3),('H2',9.3),('Ne',10.4),('He',14.6)])
    ksi=dict()
    #in m*Pa
    for a,b in k.items():
        ksi[a] = b*sc.torr*1e-3*1e-2
    return ksi[gas]/p

#Cm = lambda d,l: 121*d**3/l

def font_setup(size = 13, weight = 'normal', family = 'serif',color = 'None'):
    ''' Set-up font for Matplotlib plots
    'family':'Times New Roman','weight':'heavy','size': 18
    '''
    import matplotlib.pylab as plt
    
    font = {'family':family,'weight':weight,'size': size}
    plt.rc('font',**font)
    plt.rcParams.update({'mathtext.default':  'regular',
                         'figure.facecolor': color,
                        })

def bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
                  False - Curve is open
    """
    import scipy.interpolate as si

    # If periodic, extend the point array by count+degree+1
    cv = np.asarray(cv)
    count = len(cv)

    if periodic:
        factor, fraction = divmod(count+degree+1, count)
        cv = np.concatenate((cv,) * factor + (cv[:fraction],))
        count = len(cv)
        degree = np.clip(degree,1,degree)

    # If opened, prevent degree from exceeding count-1
    else:
        degree = np.clip(degree,1,count-1)


    # Calculate knot vector
    kv = None
    if periodic:
        kv = np.arange(0-degree,count+degree+degree-1)
    else:
        kv = np.clip(np.arange(count+degree+1)-degree,0,count-degree)

    # Calculate query range
    u = np.linspace(periodic,(count-degree),n)


    # Calculate result
    return np.array(si.splev(u, (kv,cv.T,degree))).T

def pandas_display(cols,rows):
    """ Set the number of displayed columns and rows
    for the pandas dataframe in jupyter notebook
    """
    import pandas as pd
    pd.set_option('display.max_columns',cols)
    pd.set_option('display.max_rows',rows)

def poly_smooth(x,y,order,**kws):
    """ tools.py: fit dat with a polinomial
    return array of a given length
    """
    x = np.array(x); y = np.array(y)
    xs = np.linspace(x.min(),x.max(),len(x)*kws.get('xn',10))
    fit = np.polyfit(x,y,order)
    fit_fn = np.poly1d(fit)
    return xs,fit_fn(xs)

def log_progress(sequence, every=None, size=None, name='Items'):
    """ https://github.com/alexanderkuk/log-progress
    """
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size} rec: {rec}'.format(
                        name=name,
                        index=index,
                        size=size,
                        rec=record
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index} rec:{rec} Success".format(
            name=name,
            index=str(index or '?'),
            rec=record
        )

def html_float(f,prec=2):
    """ TEXT formatting for HTML output
    """
    float_str = "{0:.{1}e}".format(f,prec)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return u"{0}\u00d710<sup>{1}</sup>".format(base, int(exponent))
    else:
        return float_str       

def html_color(s,color='red'):
    """ change text color for html output """
    return u'<font color = "{}">{}</font>'.format(color,s)

def html_snc(text,color='red',size=6,prec=2,**kws):
    """ change text color for html output """
    size = kws.get('s',size)
    try:
        text = "{0:.{1}f}".format(text,prec)
    except:
        pass
    txt = u'<font color = "{c}" size="{s}">{t}</font>'.format(c=color,t=text,s=size)
    return txt
        
if __name__ == "__main__":
    pass