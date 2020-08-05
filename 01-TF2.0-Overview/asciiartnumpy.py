import numpy as np

def array_to_asciiart_lines(image : np.array, scale:int=None):
    img= image if isinstance( image, (np.ndarray, np.generic) ) else image.numpy()
    if scale is None or scale==0:
        isint= np.max(img) > 1
        scale= 70.0/256 if isint else 70.0
    else:
        scale= 70.0/scale
    grey70="$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
    def lookup_grey70(i:int):
        return grey70[69-i]
    asGreyScale= (img * scale).astype( np.uint8)
    lines = [''.join(np.array(list(map(lookup_grey70, pixel)))) for pixel in asGreyScale]
    return lines

def print_as_asciiart(image:np.array, trim_top_and_bottom=False, scale:int=None):
    lines = array_to_asciiart_lines(image, scale=scale)
    if trim_top_and_bottom:
        lines= [ line for line in lines if line.strip() != '' ]
    print('\n'.join(lines))
    return lines