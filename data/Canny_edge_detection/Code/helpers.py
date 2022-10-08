import numpy as np
def edgeLink(M, Mag, Ori, low, high):
    # double threshold, focus on these in uncertain points
    # suppress the magnitude where edgemap is false(false means suppressed in NMS step)
    Mag_suppressed = np.where(M, Mag, 0)
    Uncertain = np.logical_and(low < Mag_suppressed, Mag_suppressed < high)
    # initial EdgeMap with high points, later add linked uncertain points to it
    EdgeMap = Mag_suppressed >= high
    # compute the edge orientation(perpendicular to the orientation direction)
    EdgeOri = Ori + np.pi / 2
    # EdgeOri = np.where(EdgeOri < 0, EdgeOri + np.pi, EdgeOri) # add pi?

    # find neighbor coordinations in the edge direction
    nr, nc = Uncertain.shape
    x, y = np.meshgrid(np.arange(nc), np.arange(nr))
    Neighbor1_x = np.clip(x + np.cos(EdgeOri), 0, nc - 1)
    Neighbor1_y = np.clip(y + np.sin(EdgeOri), 0, nr - 1)

    Neighbor2_x = np.clip(x - np.cos(EdgeOri), 0, nc - 1)
    Neighbor2_y = np.clip(y - np.sin(EdgeOri), 0, nr - 1)
    done = False

    # loop until uncertain points don't change anymore
    while not done:
        # interpolate the magnitude of neighbor
        Neighbor1 = interp2(Mag_suppressed, Neighbor1_x, Neighbor1_y)
        Neighbor2 = interp2(Mag_suppressed, Neighbor2_x, Neighbor2_y)
        
        # if its neighbor is above high threshold, then it has a strong point neighbor
        # copy the magnitude and remove it from uncertain points
        nearStrongPoints = np.logical_or(Neighbor1 >= high,
                                         Neighbor2 >= high)
        toUpdate = np.logical_and(Uncertain, nearStrongPoints)
        # update the magnitude of uncertain-near-strong points
        Mag_suppressed = np.where(toUpdate, np.maximum(Neighbor1, Neighbor2), Mag_suppressed)
        Uncertain[toUpdate] = False
        EdgeMap[toUpdate] = True
        if not np.any(toUpdate):
            done = True
            
    return EdgeMap


def interp2(v, xq, yq):

	if len(xq.shape) == 2 or len(yq.shape) == 2:
		dim_input = 2
		q_h = xq.shape[0]
		q_w = xq.shape[1]
		xq = xq.flatten()
		yq = yq.flatten()

	h = v.shape[0]
	w = v.shape[1]
	if xq.shape != yq.shape:
		raise 'query coordinates Xq Yq should have same shape'

	x_floor = np.floor(xq).astype(np.int32)
	y_floor = np.floor(yq).astype(np.int32)
	x_ceil = np.ceil(xq).astype(np.int32)
	y_ceil = np.ceil(yq).astype(np.int32)

	x_floor[x_floor < 0] = 0
	y_floor[y_floor < 0] = 0
	x_ceil[x_ceil < 0] = 0
	y_ceil[y_ceil < 0] = 0

	x_floor[x_floor >= w-1] = w-1
	y_floor[y_floor >= h-1] = h-1
	x_ceil[x_ceil >= w-1] = w-1
	y_ceil[y_ceil >= h-1] = h-1

	v1 = v[y_floor, x_floor]
	v2 = v[y_floor, x_ceil]
	v3 = v[y_ceil, x_floor]
	v4 = v[y_ceil, x_ceil]

	lh = yq - y_floor
	lw = xq - x_floor
	hh = 1 - lh
	hw = 1 - lw

	w1 = hh * hw
	w2 = hh * lw
	w3 = lh * hw
	w4 = lh * lw

	interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

	if dim_input == 2:
		return interp_val.reshape(q_h, q_w)
	return interp_val