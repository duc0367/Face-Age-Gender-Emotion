def convert_tl_br_to_tl_wh(tl: tuple, br: tuple) -> tuple:
    width = br[0] - tl[0]
    height = br[1] - tl[1]
    return tl, (width, height)
