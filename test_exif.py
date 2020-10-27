# encoding=utf-8

# from exif import EXIF
import exifread
SENSOR = os.path.join(abspath, 'data', 'sensor_data.json')


def open_rt(path):
    """Open a file in text mode for reading utf-8."""
    return open(path, 'r', encoding='utf-8')


def json_load(fp):
    return json.load(fp)


def json_loads(text):
    return json.loads(text)


def _decode_make_model(value):
    """Python 2/3 compatible decoding of make/model field."""
    if hasattr(value, 'decode'):
        try:
            return value.decode('utf-8')
        except UnicodeDecodeError:
            return 'unknown'
    else:
        return value


def eval_frac(value):
    try:
        return float(value.num) / float(value.den)
    except ZeroDivisionError:
        return None


def extract_make(tags):
    # Camera make and model
    if 'EXIF LensMake' in tags:
        make = tags['EXIF LensMake'].values
    elif 'Image Make' in tags:
        make = tags['Image Make'].values
    else:
        make = 'unknown'
    return _decode_make_model(make)


def extract_model(tags):
    if 'EXIF LensModel' in tags:
        model = self.tags['EXIF LensModel'].values
    elif 'Image Model' in tags:
        model = tags['Image Model'].values
    else:
        model = 'unknown'
    return _decode_make_model(model)


def extract_image_size(tags):
    # Image Width and Image Height
    if ('EXIF ExifImageWidth' in tags and  # PixelXDimension
            'EXIF ExifImageLength' in tags):  # PixelYDimension
        width, height = (int(tags['EXIF ExifImageWidth'].values[0]),
                         int(tags['EXIF ExifImageLength'].values[0]))
    elif ('Image ImageWidth' in tags and
          'Image ImageLength' in tags):
        width, height = (int(tags['Image ImageWidth'].values[0]),
                         int(tags['Image ImageLength'].values[0]))
    else:
        width, height = -1, -1
    return width, height


def sensor_string(make, model):
    if make != 'unknown':
        # remove duplicate 'make' information in 'model'
        model = model.replace(make, '')
    return (make.strip() + ' ' + model.strip()).strip().lower()


def get_mm_per_unit(self, resolution_unit):
    """Length of a resolution unit in millimeters.

    Uses the values from the EXIF specs in
    https://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/EXIF.html

    Args:
        resolution_unit: the resolution unit value given in the EXIF
    """
    if resolution_unit == 2:    # inch
        return inch_in_mm
    elif resolution_unit == 3:  # cm
        return cm_in_mm
    elif resolution_unit == 4:  # mm
        return 1
    elif resolution_unit == 5:  # um
        return um_in_mm
    else:
        logger.warning(
            'Unknown EXIF resolution unit value: {}'.format(resolution_unit))
        return None


def get_tag_as_float(tags, key, index=0):
    if key in tags:
        val = tags[key].values[index]
        if isinstance(val, exifread.utils.Ratio):
            ret_val = eval_frac(val)
            if ret_val is None:
                logger.error(
                    'The rational "{2}" of tag "{0:s}" at index {1:d} c'
                    'aused a division by zero error'.format(
                        key, index, val))
            return ret_val
        else:
            return float(val)
    else:
        return None


def extract_sensor_width(tags):
    """Compute sensor with from width and resolution."""
    if ('EXIF FocalPlaneResolutionUnit' not in tags or
            'EXIF FocalPlaneXResolution' not in tags):
        return None

    resolution_unit = tags['EXIF FocalPlaneResolutionUnit'].values[0]
    mm_per_unit = get_mm_per_unit(resolution_unit)
    if not mm_per_unit:
        return None

    pixels_per_unit = get_tag_as_float(tags, 'EXIF FocalPlaneXResolution')
    if pixels_per_unit <= 0:
        pixels_per_unit = get_tag_as_float(
            tags, 'EXIF FocalPlaneYResolution')
        if pixels_per_unit <= 0:
            return None

    units_per_pixel = 1 / pixels_per_unit
    width_in_pixels = extract_image_size()[0]

    return width_in_pixels * units_per_pixel * mm_per_unit


def compute_focal(focal_35, focal, sensor_width, sensor_string):
    if focal_35 is not None and focal_35 > 0:
        focal_ratio = focal_35 / 36.0  # 35mm film produces 36x24mm pictures.
    else:
        if not sensor_width:
            sensor_width = sensor_data.get(sensor_string, None)
        if sensor_width and focal:
            focal_ratio = focal / sensor_width
            focal_35 = 36.0 * focal_ratio
        else:
            focal_35 = 0
            focal_ratio = 0
    return focal_35, focal_ratio


def extract_focal(tags):
    make, model = extract_make(tags), extract_model(tags)
    focal_35, focal_ratio = compute_focal(
        get_tag_as_float(tags, 'EXIF FocalLengthIn35mmFilm'),
        get_tag_as_float(tags, 'EXIF FocalLength'),
        extract_sensor_width(tags),
        sensor_string(make, model))
    return focal_35, focal_ratio


file_path = 'f:/tmp/00000.jpg'
with open(file_path, 'rb') as f_h:
    # exif = EXIF(f_h)

    # focal_35, focal_ratio = exif.extract_focal()
    # print(focal_35)
    # print(focal_ratio)

    # -----
    tags = exifread.process_file(f_h, details=False)
    make, model = extract_make(tags), extract_model(tags)
    print(make)
    print(model)

    focal_35, focal_ratio = extract_focal(tags)
    print(focal_35)
    print(focal_ratio)

    with open_rt(SENSOR) as f:
        sensor_data = json_load(f)


print('Done.')
