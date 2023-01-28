import importlib
import argparse
import gc
import math
import os
import glob
import cv2
import numpy as np
import subprocess
import random
import tempfile 
import io
from enum import Enum

from PIL import Image, ImageFilter, ImageCms
from tqdm import tqdm
import shutil 

class Snap(Enum):
    NONE = 0
    TOP_LEFT = 1
    TOP = 2
    TOP_RIGHT = 3
    MID_LEFT = 4
    MID = 5
    MID_RIGHT = 6
    BOT_LEFT = 7
    BOT = 8
    BOT_RIGHT = 9

args = None

#=======================================================================================
def filenameToPNG(image_path):
    return os.path.splitext(os.path.basename(image_path))[0] + ".png"

def ImageExpandCanvas(image, w , h, color=(255, 255, 255, 255) , snap = Snap.MID ):    
    if image.width == w and image.height == h:
        return image

    background = Image.new('RGB', (w , h), color)
    #if hasattr( image, "info") and hasattr( image.info, "icc_profile"):
        #background.info['icc_profile'] = image.info['icc_profile']
    if snap == Snap.NONE:
        snap = snap.MID

    px = 0
    py = 0
    if snap == Snap.MID or snap == Snap.MID_LEFT or snap == snap.MID_RIGHT :        
        py = int(( h - image.height ) / 2)
    elif snap == Snap.BOT or snap == Snap.BOT_LEFT or snap == snap.BOT_RIGHT :
        py = ( h - image.height )

    if snap == Snap.MID or snap == Snap.TOP or snap == snap.BOT :        
        px = int(( w - image.width ) / 2)
    elif snap == Snap.TOP_RIGHT or snap == Snap.MID_RIGHT or snap == snap.BOT_RIGHT :
        px = ( w - image.width )

    pos = (px , py)
    if image.mode == "RGB":
        background.paste(image, pos)  # 3 is the alpha channel
    else:
        background.paste(image, pos, mask=image.split()[3])  # 3 is the alpha channel
    return background

def ImageConvertToRGB(image, color=(255, 255, 255, 255)):    
    if image.mode == "RGB":
        return image, False
    else:        
        background = Image.new('RGB', image.size, color)
        #if hasattr( image, "info") and hasattr( image.info, "icc_profile"):
            #background.info['icc_profile'] = image.info['icc_profile']

        background.paste(image, mask=image.split()[3])        
        return background, True

def image_crop_color(img, c):
    np_array = np.array(img)    
    mask = np_array != c
    coords = np.argwhere(mask)
    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0) + 1    
    cropped_box = np_array[x0:x1, y0:y1, z0:z1]    
    img_crop = Image.fromarray(cropped_box, img.mode)
    return img_crop

def image_crop_transparent(img):        
    if img.mode == "RGB":
        return img    
    img = image_crop_color(img, [255, 255, 255, 0])
    img = image_crop_color(img, [0, 0, 0, 0])
    return img

def image_crop_corner(img):    
    img = image_crop_color(img, img.getpixel((0,0)))
    img = image_crop_color(img, img.getpixel((img.width-1,0)))
    img = image_crop_color(img, img.getpixel((0,img.height-1)))
    img = image_crop_color(img, img.getpixel((img.width-1,img.height-1)))
    return img

def image_downscale(image, w, h):
    image = image.resize( (w, h), resample=Image.BICUBIC )
    image = image.filter(ImageFilter.DETAIL)

    #image = image.resize( (w, h), resample=Image.BICUBIC )
    #image = image.filter(ImageFilter.SHARPEN)

    #image = numpy.array(image)
    #image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)#cv2.INTER_LANCZOS4)#cv2.INTER_AREA)
    #image = Image.fromarray(image)
    return image


def image_apply_icc_profile(image):    
    if image.info is not None :            
        icc = image.info.get('icc_profile')    
        if icc:
            image = ImageCms.profileToProfile(image, ImageCms.ImageCmsProfile(io.BytesIO(icc)), ImageCms.createProfile('sRGB'))
    return image


#=======================================================================================

#=======================================================================================
def get_resolutions(max_reso, min_size=256, max_size=1024, tile_size=64):
    max_width, max_height = max_reso
    max_tile = (max_width // tile_size) * (max_height // tile_size)

    resos = set()

    #add quad
    size = int(math.sqrt(max_tile)) * tile_size
    resos.add((size, size, size * size))
    
    size = min_size
    while size <= max_size:
        width = size
        height = min(max_size, (max_tile // (width // tile_size)) * tile_size)
        if height < max_height:
            resos.add((int(width), int(height), int(width) * int(height)))
        if width < max_width:
            resos.add((int(height), int(width), int(width) * int(height)))

        size += tile_size
    
    resos = list(resos)
    resos.sort()

    return resos


#SD의 편향 학습을 교란시키기 위해, 캔버스를 확장할 때 가능하면 이미지에 존재하지 않는 랜덤색으로 채운다.
def get_random_color_for_expand(img):
    np_array = np.array(img)   
    
    i_step = 0
    i_thresh = 50
    while i_step < 300:        
        c = ( random.randint(0,255), random.randint(0,255), random.randint(0,255))

        c1 = ( c[0] - i_thresh , c[1] - i_thresh , c[2] - i_thresh)        
        c2 = ( c[0] + i_thresh , c[1] + i_thresh , c[2] + i_thresh)        
        b = cv2.inRange(np_array, c1, c2).any() 
        if b == False:
            break
        i_step += 1        
        i_thresh = max( 1 , i_thresh - 1 )                
    
    '''Note: 아래는 밝은 색을 우선해서 Canvas를 늘리는 방법. 폐기됨.
    학습 세트가 전체적으로 밝은 경우, 이것 또한 자체적인 편향성을 갖게 되어 랜덤칼라 캔버스의 의도가 퇴색된다'''
    #MIN_REDUCE = 5    
    #i_min = 230 #어두운 색일 수록 학습 영향력이 크므로, 가능하면 밝은 색부터    
    #i_thresh:int = ( i_min // MIN_REDUCE ) // 2
    #i_step = 0
    #while i_step < 300:        
        #c = ( random.randint(i_min,255), random.randint(i_min,255), random.randint(i_min,255))
        #c1 = ( c[0] - i_thresh , c[1] - i_thresh , c[2] - i_thresh)        
        #c2 = ( c[0] + i_thresh , c[1] + i_thresh , c[2] + i_thresh)        
        #b = cv2.inRange(np_array, c1, c2).any() 
        #if b == False:
            #break
        #i_step += 1
        #i_min = max(0 , i_min - MIN_REDUCE)
        #i_thresh = max( 1 , i_thresh - 1 )                
    return c


def add_size_to_suggest(sizes, reso, image):
    max_ = max(image.width , image.height)
    min_ = min(image.width , image.height)

    if reso[0] > max_ or reso[1] > min_:
        return

    for r_prev in enumerate(sizes):
        if r_prev[0] == reso[0] and r_prev[1] == reso[1]:
            return

    f = reso[0] / max_
    sizes.add( (reso[0], reso[1], reso[2] - int( int(image.width * f ) * int(image.height * f) ) , True ) )
    f = reso[1] / min_                    
    sizes.add( (reso[0], reso[1], reso[2] - int( int(image.width * f ) * int(image.height * f) ) , False ) )
    return

def image_adjust_size_for_SD(image, image_path, resolutions, expand_snap):
        
    ratio = image.width / image.height

    if args.random_canvas_color == False:
        color_expand_canvas = (255, 255, 255, 255)        
    else:
        color_expand_canvas = get_random_color_for_expand( image )

    res_target = None        
    for i, reso in enumerate(resolutions):            
        if reso[0] == image.width and reso[1] == image.height:
            res_target = reso
            break
        if reso[1] == image.width and reso[0] == image.height:
            res_target = reso
            break
    if res_target is None:            
        for i, res in enumerate(resolutions):            
            if res[0] / res[1] == ratio or res[1] / res[0] == ratio :
                res_target = res
                break
    
    if res_target is None:
        
        #추천 해상도를 찾는다.
        sizes = set()

        #Crop 해상도, Expand 해상도 구하기
        px_count = image.width * image.height
        max_ = max(image.width , image.height)
        min_ = min(image.width , image.height)


        #양쪽 기준 Expand 해상도
        reso_simple_expand = None
        i_mx = (int(max_//64) + 1) * 64
        i_mn = (int(min_//64) + 1) * 64
        for i, reso in reversed(list(enumerate(resolutions))):
            if reso[0] == i_mx and reso[1] == i_mn:                
                add_size_to_suggest(sizes , reso, image)
                reso_simple_expand = reso
                mtd = "\033[92m Expand \033[0m"
                if ratio > 1:
                    new_w = reso[0]
                    new_h = reso[1]                                        
                else:
                    new_w = reso[1]
                    new_h = reso[0]
                print(f"    {mtd} Suggest \033[92m{new_w}\033[0m, \033[92m{new_h}\033[0m \033[96m Simply, Canvas Expand\033[0m or \033[31mUPScale\033[0m \t(diff {reso[2] - px_count})")
                

        #큰쪽 기준 근접 해상도
        for i, reso in reversed(list(enumerate(resolutions))):
            if reso[0] > max_:
                continue            
            f = reso[0] / max_
            m = min_ * f
            i_d = (m//64) * 64                
            
            for i2, reso2 in enumerate(resolutions):
                if reso2[0] != reso[0]:
                    continue
                if i_d == reso2[1]:             
                    #print(f"a {reso[0]}")                                  
                    add_size_to_suggest( sizes , reso2, image)                                         
                if i_d + 64 == reso2[1]:
                    #reso2 = bucket_resos[i2 + 1]                    
                    #print(f"b {reso} , {reso2}")                       
                    add_size_to_suggest(sizes , reso2, image)
                

        #작은쪽 기준으로 근접 해상도
        for i, reso in enumerate(resolutions):                                                    
            if reso[0] / reso[1] > max_ / min_ and reso[0] <= max_ and reso[1] <= min_:
                add_size_to_suggest( sizes, reso, image )
                if i > 0 :
                    reso_temp = resolutions[i - 1]                                                                      
                    add_size_to_suggest( sizes , reso_temp, image)
                break
            
        sizes = list(sizes)
        sizes.sort( key=lambda child: child[2], reverse=False )
        
        #if image.width * image.height < sizes[2][2]:
            #print(f"    \033[31mNeed UPSCALE!!\033[0m ( {image.width*image.height} ) ")
        if reso_simple_expand is not None and args.expand == True:                
            if args.dir_out is not None :                    
                if ratio > 1:
                    new_w = reso_simple_expand[0]
                    new_h = reso_simple_expand[1]                                        
                else:
                    new_w = reso_simple_expand[1]
                    new_h = reso_simple_expand[0]

                print(f"    \033[92m Canvas expanded Simply {image.size} to ({new_w}, {new_h})\033[0m")
                image_new:Image = ImageExpandCanvas( image , new_w, new_h, color_expand_canvas, expand_snap )
                image_new.save( os.path.join(args.dir_out, filenameToPNG(image_path)), "png")   
            return 21 #count_copy +=1
        elif len(sizes) < 1 :
            print("    \033[31mNeed UPScale\033[0m")
            return 91 #count_need_upscale += 1
        else:
            b_saved = False
            reso_resize = None
            for i, reso in enumerate(sizes):                
                if ratio > 1:
                    new_w = reso[0]
                    new_h = reso[1]
                    mtd_by = "\033[95mW\033[0m" if reso[3] == True else "\033[96mH\033[0m"  
                    b_resize_by_w = (reso[3] == True)            
                else:
                    new_w = reso[1]
                    new_h = reso[0]
                    mtd_by = "\033[96mH\033[0m" if reso[3] == True else "\033[95mW\033[0m"
                    b_resize_by_w = (reso[3] != True)
                    
                if b_resize_by_w == True:
                    org_w = image.width #resize with width
                    org_h = int( new_h * ( image.width / new_w ) )
                    s_org_w = f"{org_w}"
                    s_org_h = f"\033[92m{org_h}\033[0m"
                else:
                    org_h = image.height #resize with width
                    org_w = int( new_w * ( image.height / new_h ) )
                    s_org_h = f"{org_h}"
                    s_org_w = f"\033[92m{org_w}\033[0m"


                if reso[2] < 0 :                    
                    mtd = "\033[31m Crop.. \033[0m"                    
                else:
                    mtd = "\033[92m Expand \033[0m"

                print(f"    {mtd} Suggest \033[92m{new_w}\033[0m, \033[92m{new_h}\033[0m (Resize {mtd_by})\tCanvas To {s_org_w} x {s_org_h}\t(diff {reso[2]})")

                if args.expand == True:
                    if reso[2] > 0 and reso_resize is None:
                        reso_resize = reso
                else:
                    if reso[2] <= 0 :
                        reso_resize = reso

               

            if b_saved == False and reso_resize is not None:                    
                b_saved = True
                if args.dir_out is not None :
                    if ratio > 1:
                        new_w = reso_resize[0]
                        new_h = reso_resize[1]                        
                    else:
                        new_w = reso_resize[1]
                        new_h = reso_resize[0]                        

                    if reso_resize[3] == True:
                        if ratio > 1:
                            w = new_w
                            h = int(image.height * ( w / image.width  ))
                        else:
                            h = new_h
                            w = int(image.width * ( h / image.height  ))
                    else:
                        if ratio > 1:
                            h = new_h
                            w = int(image.width * ( h / image.height  ))                                
                        else:
                            w = new_w
                            h = int(image.height * ( w / image.width ))     
                    
                    if new_w < image.width or new_h < image.height:
                        print(f"    \033[92m Downscale + Fill ( {new_w} x {new_h} )\033[0m")
                    else:
                        print(f"    \033[92m Resize + Fill ( {new_w} x {new_h} )\033[0m")
                    
                    image = image_downscale(image , w,h)                        
                    image_new:Image = ImageExpandCanvas( image , new_w, new_h, color_expand_canvas, expand_snap )
                    image_new.save( os.path.join(args.dir_out, filenameToPNG(image_path)), "png")                    
            if b_saved:
                return 21 #count_copy +=1
            else:
                return 92 #count_no_res += 1            
    else:            
        if ratio > 1 :
            nw = res_target[0]
            nh = res_target[1]
        else:
            nw = res_target[1]
            nh = res_target[0]
        
        if image.width < nw or image.height < nh:
            print("    \033[31mNeed UPScale\033[0m")
            return 91 #count_need_upscale += 1        
        elif nw == image.width and nh == image.height: 
            print(f"(Fit) {os.path.basename(image_path)}. image size={image.width} x {image.height}\033[0m")                        
            if args.dir_out is not None:
                shutil.copy( image_path , os.path.join(args.dir_out, os.path.basename(image_path)) )
            return 0
        elif args.dir_out is not None:
            print(f"(Fit with Resize) {os.path.basename(image_path)}. ({image.width} x {image.height}) TO ({nw} x {nh})\033[0m")
            if image.width != nw or image.height != nh:
                image = image_downscale(image , nw, nh)
                image = ImageExpandCanvas( image , nw, nh, color_expand_canvas, expand_snap )
            image.save( os.path.join(args.dir_out, filenameToPNG(image_path) ), "png")
            return 0
        else:
            print(f"\033[95m(Fit but need resize)\033[0m {os.path.basename(image_path)}. image size={image.width} x {image.height} to \033[92m{nw}\033[0m x \033[92m{nh}\033[0m")
            return 11
#=======================================================================================

def subprocess_upscale(s_path_in, s_path_out, denoise_level = 0):
    subprocess.run(["python", "-m", "waifu2x.cli", "-n", str(denoise_level), "--tta", "-m", "noise_scale", "-i" , s_path_in, "-o", s_path_out])
    return

def main(args):
    
    res:int = args.res
    res_min:int = args.res_min
    resolutions = get_resolutions( (res,res) , res_min, res / res_min * res  )

    #print(f"=======================================================")
    #for i, reso in enumerate(resolutions):
        #print(f"Resolution{i}\t({reso[0]}x{reso[1]}\t={reso[2]})")
    #print(f"=======================================================")
    
    image_paths = \
        glob.glob(os.path.join(args.dir_in, "*.jpg")) + \
        glob.glob(os.path.join(args.dir_in, "*.jpeg")) + \
        glob.glob(os.path.join(args.dir_in, "*.png")) + \
        glob.glob(os.path.join(args.dir_in, "*.webp"))

    image_paths.sort(key=lambda v: v.upper(), reverse=True)
    
    count_fit = 0
    count_resize = 0
    count_resize_fill = 0
    count_need_upscale = 0
    count_no_res = 0
    
    if args.expand_snap is None:
        if args.expand == True:
            expand_snap = Snap.BOT
        else:
            expand_snap = Snap.TOP
    else:
        expand_snap = Snap[args.expand_snap.upper()]

    files = set()

    for i, image_path in enumerate(image_paths):                        
        image:Image = Image.open(image_path)
        image.load()
        image = image_apply_icc_profile( image )
        
        image_org_size = image.size
        print(f"\033[95m{i + 1}/{len(image_paths)}] {os.path.basename(image_path)}")
        b_cropped_tranparent = False
        image = image_crop_transparent(image)
        if image.size != image_org_size:
            print(f"\tCrop (Transparent Pixels) {image_org_size} to {image.size}")
            b_cropped_tranparent = True      

        image, b_converted_RGB = ImageConvertToRGB(image)

        image_size_prev_crop_corner = image.size
        image = image_crop_corner(image)        
        if image.size != image_size_prev_crop_corner:
            print(f"\tCrop (Corner Color) {image_org_size} to {image.size}")

        #Save if Cropped
        if image.size != image_org_size or b_converted_RGB:            
            image_path = os.path.join(args.dir_in_crop, filenameToPNG(image_path))
            image.save( image_path, "png" )
            
        if image.size != image_org_size:
            print(f"\033[95m\tPure size={image_org_size}. Cropped size={image.size}\033[0m")

        i_prevent_loop = 0
        while i_prevent_loop < 10:
            i_prevent_loop += 1
            result = image_adjust_size_for_SD(image, image_path, resolutions, expand_snap)
            if result == 91 and args.dir_in_upscale is not None:
                image.close()

                if os.path.exists(os.path.join( args.dir_in_upscale )) == False:
                    os.makedirs( os.path.join( args.dir_in_upscale ), exist_ok=True)
                print(f"    Upscaling by waifu2x ...{image.size} to ({image.width * 2},{image.height * 2})")
                image_path_upscale = os.path.join(args.dir_in_upscale , os.path.basename(image_path))
                subprocess_upscale( image_path, image_path_upscale )     
                #replace image to upscaled           
                image_path = image_path_upscale
                image = Image.open(image_path)
                image.load()
                image = image_apply_icc_profile( image )
            else:
                break
        image.close()
        
        files.add( ( os.path.basename(image_path), result ))

        if result == 0:
            count_fit += 1
        elif result == 11:
            count_resize += 1
        elif result == 21:
            count_resize_fill += 1
        elif result == 91:
            if args.dir_in_upscale is None:
                count_need_upscale += 1
        else:
             count_no_res += 1
       
    print(f"=======================================================")
    if count_fit > 0:
        print(f"Fit \033[92m{count_fit}\033[0m")    
    if count_resize > 0:
        print(f"Resize \033[95m{count_resize}\033[0m")
    if count_resize_fill > 0:
        print(f"ResizeFill \033[31m{count_resize_fill}\033[0m")
    if count_need_upscale > 0:
        print(f"UPScale \033[31m{count_need_upscale}\033[0m")
    if count_no_res > 0:
        print(f"No Res \033[31m{count_no_res}\033[0m")

    if args.clear_temp == True:
        shutil.rmtree(args.dir_in_crop)
        shutil.rmtree(args.dir_in_upscale)

    return

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir_in", type=str, default=None, help="Directory for check", required=True)    
    parser.add_argument("--dir_out", type=str, default=None, help="Directory for output")
    parser.add_argument("--res", type=int, default=1024, help="Size")
    parser.add_argument("--res_min", type=int, default=256, help="Size")

    parser.add_argument("--dir_in_crop", type=str, default=None, help="Temporary Directory for cropped image save")
    parser.add_argument("--dir_in_upscale", type=str, default=None, help="Temporary Directory for upscaled image save")

    parser.add_argument("--expand", default=True, type=str2bool, help="Expand canvas to suggested size. False to crop")
    parser.add_argument("--expand_snap", default=None, type=str , choices=["TOP_LEFT", "TOP", "TOP_RIGHT", "MID_LEFT", "MID", "MID_RIGHT", "BOT_LEFT", "BOT", "BOT_RIGHT"] , help="Expand canvas from Postion")
    parser.add_argument("--random_canvas_color", default=False, type=str2bool, help="Expands canvas to random color")

    parser.add_argument("--clear_temp", default=True, type=str2bool, help="Clear temporary files (crop, upscale)")

    args = parser.parse_args()

    if args.dir_in_crop is None:
        args.dir_in_crop = os.path.join( args.dir_in , "in_crop" )
        if os.path.exists(os.path.join(args.dir_in_crop)) == False:
            os.makedirs( os.path.join(args.dir_in_crop), exist_ok=True)
    
    if args.dir_in_upscale is None:
        args.dir_in_upscale = os.path.join( args.dir_in , "in_upscale" )
        if os.path.exists(os.path.join(args.dir_in_upscale)) == False:
            os.makedirs(os.path.join(args.dir_in_upscale), exist_ok=True)

    if args.dir_out is None:
        args.dir_out = os.path.join( args.dir_in , "out" )
        if os.path.exists(os.path.join(args.dir_out)) == False:
            os.makedirs(os.path.join(args.dir_out), exist_ok=True)

    main(args)