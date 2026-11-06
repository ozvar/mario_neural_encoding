#!/usr/bin/env Rscript
# Dual-band surface map replicating scripts/python/plot_dual_band_map.py.
# Renders 7 PNGs: flatmap LH/RH, inflated LH/RH x lateral/medial, legend.
# Reads pre-extracted vertex arrays from results/r_cifti_arrays/ (written by
# scripts/python/export_cifti_vertex_arrays.py).

# -- Early display setup -------------------------------------------------------
# Must run before library(rgl) connects to X11.  If the current display is
# smaller than the required render window, start an Xvfb virtual display.
local({
  args <- commandArgs(trailingOnly = TRUE)
  dpi_idx <- which(args == "--dpi")
  dpi <- if (length(dpi_idx) == 1L && dpi_idx < length(args)) {
    as.integer(args[dpi_idx + 1L])
  } else 300L
  w <- as.integer(round(1920 * dpi / 96))
  h <- as.integer(round(1080 * dpi / 96))
  dpyinfo <- tryCatch(system2("xdpyinfo", stdout = TRUE, stderr = FALSE), error = function(e) character(0))
  dim_line <- grep("dimensions:", dpyinfo, value = TRUE)[1]
  m <- if (!is.na(dim_line)) regmatches(dim_line, regexpr("[0-9]+x[0-9]+", dim_line)) else character(0)
  need_vfb <- TRUE
  if (length(m) == 1L) {
    parts <- as.integer(strsplit(m, "x")[[1]])
    need_vfb <- parts[1] < w || parts[2] < h
  }
  if (need_vfb) {
    vdisp <- ":98"
    system2("pkill", c("-f", paste0("Xvfb ", vdisp)), stdout = FALSE, stderr = FALSE)
    Sys.sleep(0.3)
    system2("Xvfb", c(vdisp, "-screen", "0", sprintf("%dx%dx24", w + 10L, h + 10L)),
            wait = FALSE, stdout = FALSE, stderr = FALSE)
    Sys.sleep(1.5)
    Sys.setenv(DISPLAY = vdisp)
    message("  Xvfb ", vdisp, " started at ", w, "x", h)
  }
})

suppressPackageStartupMessages({
  library(optparse)
  library(gifti)
  library(rgl)
  library(png)
  library(grDevices)
})

# -- CLI -----------------------------------------------------------------------
option_list <- list(
  make_option("--group",        default = "GROUP_N5_PRIMARY"),
  make_option("--vmax",         type = "double", default = 0.20,
              help = "PM colormap ceiling [default %default]"),
  make_option("--pm-threshold", type = "double", default = 0.0,
              help = "Magnitude floor added to significance gate [default %default]"),
  make_option("--views",        default = "all",
              help = "all | flatmap | inflated [default %default]"),
  make_option("--dpi",          type = "integer", default = 300L,
              help = "Output DPI; scales render window from 96-DPI baseline [default %default]")
)
opt <- parse_args(OptionParser(option_list = option_list))

# Derive pixel dimensions from DPI. Baseline is 1920x1080 at 96 DPI.
BASE_DPI <- 96L
render_width  <- as.integer(round(1920 * opt$dpi / BASE_DPI))
render_height <- as.integer(round(1080 * opt$dpi / BASE_DPI))

# -- Paths ---------------------------------------------------------------------
REPO <- "/home/ozvar/Git/cneuromod/mario_neural_encoding"

GROUP_LABELS <- list(
  GROUP_N5_PRIMARY = "group_n5_primary_analysis",
  GROUP_N3_PRIMARY = "group_n3_primary_analysis"
)
label <- GROUP_LABELS[[opt$group]]
if (is.null(label)) stop("Unknown --group: ", opt$group)

ARRAY_DIR <- file.path(REPO, "results/r_cifti_arrays", label)
OUT_DIR    <- file.path(REPO, "results/figures/r_ciftitools", label)
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

LUT_PATH   <- path.expand("~/.conda/envs/mario/share/pycortex/colormaps/mario_dual_band.png")
SURF_L_INF  <- "/home/ozvar/Atlases/HCP_S1200_Atlas/S1200.L.inflated_MSMAll.32k_fs_LR.surf.gii"
SURF_R_INF  <- "/home/ozvar/Atlases/HCP_S1200_Atlas/S1200.R.inflated_MSMAll.32k_fs_LR.surf.gii"
SURF_L_FLAT <- "/home/ozvar/Atlases/HCP_S1200_Atlas/S1200.L.flat.32k_fs_LR.surf.gii"
SURF_R_FLAT <- "/home/ozvar/Atlases/HCP_S1200_Atlas/S1200.R.flat.32k_fs_LR.surf.gii"

vmax      <- opt$vmax
pm_thresh <- opt[["pm-threshold"]]

message("Group:  ", opt$group, "  (", label, ")")
message("DPI:    ", opt$dpi, "  render: ", render_width, "x", render_height)
message("vmax:   ", vmax, "  pm_threshold: ", pm_thresh)
message("Output: ", OUT_DIR)

# -- .npy readers --------------------------------------------------------------
.npy_header <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con))
  readBin(con, "raw", n = 6)                             # magic
  version       <- readBin(con, "raw", n = 2)
  hdr_len_bytes <- if (version[1] < 2L) 2L else 4L
  hdr_len       <- readBin(con, "integer", n = 1L, size = hdr_len_bytes,
                            endian = "little", signed = FALSE)
  rawToChar(readBin(con, "raw", n = hdr_len))
}

.npy_shape <- function(hdr) {
  m <- regmatches(hdr, regexpr("'shape':[[:space:]]*[(]([0-9, ]+)[)]", hdr))
  as.integer(unlist(regmatches(m, gregexpr("[0-9]+", m))))
}

# Read 1-D float32/float64 array.
load_npy <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con))
  readBin(con, "raw", n = 6)
  version       <- readBin(con, "raw", n = 2)
  hdr_len_bytes <- if (version[1] < 2L) 2L else 4L
  hdr_len       <- readBin(con, "integer", n = 1L, size = hdr_len_bytes,
                            endian = "little", signed = FALSE)
  hdr           <- rawToChar(readBin(con, "raw", n = hdr_len))
  n_elem        <- prod(.npy_shape(hdr))
  sz            <- if (grepl("float32", hdr)) 4L else 8L
  readBin(con, "double", n = n_elem, size = sz, endian = "little")
}

# Detect numpy dtype from header string; returns list(type, size).
.npy_dtype <- function(hdr) {
  if (grepl("'<i4'", hdr, fixed = TRUE) || grepl("<i4",  hdr, fixed = TRUE)) {
    list(type = "integer", size = 4L)
  } else if (grepl("'<f4'", hdr, fixed = TRUE) || grepl("<f4", hdr, fixed = TRUE)) {
    list(type = "double",  size = 4L)
  } else {
    list(type = "double",  size = 8L)   # float64 default
  }
}

# Read 2-D (N x C) array; supports float32, float64, int32.
load_npy_2d <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con))
  readBin(con, "raw", n = 6)
  version       <- readBin(con, "raw", n = 2)
  hdr_len_bytes <- if (version[1] < 2L) 2L else 4L
  hdr_len       <- readBin(con, "integer", n = 1L, size = hdr_len_bytes,
                            endian = "little", signed = FALSE)
  hdr           <- rawToChar(readBin(con, "raw", n = hdr_len))
  dims          <- .npy_shape(hdr)
  dt            <- .npy_dtype(hdr)
  vals <- readBin(con, dt$type, n = prod(dims), size = dt$size, endian = "little",
                  signed = TRUE)
  matrix(vals, nrow = dims[1], ncol = dims[2], byrow = TRUE)
}

# Read 1-D int32 array.
load_npy_int <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con))
  readBin(con, "raw", n = 6)
  version       <- readBin(con, "raw", n = 2)
  hdr_len_bytes <- if (version[1] < 2L) 2L else 4L
  hdr_len       <- readBin(con, "integer", n = 1L, size = hdr_len_bytes,
                            endian = "little", signed = FALSE)
  hdr           <- rawToChar(readBin(con, "raw", n = hdr_len))
  n_elem        <- prod(.npy_shape(hdr))
  readBin(con, "integer", n = n_elem, size = 4L, endian = "little", signed = TRUE)
}

message("\nLoading vertex arrays...")
motor_L <- load_npy(file.path(ARRAY_DIR, "motor_pm_L.npy"))
motor_R <- load_npy(file.path(ARRAY_DIR, "motor_pm_R.npy"))
scene_L <- load_npy(file.path(ARRAY_DIR, "scene_pm_L.npy"))
scene_R <- load_npy(file.path(ARRAY_DIR, "scene_pm_R.npy"))
msig_L  <- load_npy(file.path(ARRAY_DIR, "motor_sig_L.npy"))
msig_R  <- load_npy(file.path(ARRAY_DIR, "motor_sig_R.npy"))
ssig_L  <- load_npy(file.path(ARRAY_DIR, "scene_sig_L.npy"))
ssig_R  <- load_npy(file.path(ARRAY_DIR, "scene_sig_R.npy"))

SULC_DIR <- file.path(REPO, "results/r_cifti_arrays")
sulc_L <- load_npy(file.path(SULC_DIR, "sulc_L.npy"))
sulc_R <- load_npy(file.path(SULC_DIR, "sulc_R.npy"))

# -- Load network border indices / weights -------------------------------------
BORDER_DIR <- file.path(REPO, "results/r_cifti_arrays/borders")

BORDER_COLORS <- list(
  visual2           = "#6400ff",   # Cole-Anticevic: (0.392,0,1)
  cingulo_opercular = "#990099"    # Cole-Anticevic: (0.6,0,0.6)
)
BORDER_LWD <- 10.0

borders <- list()
for (net in names(BORDER_COLORS)) {
  for (hemi in c("L", "R")) {
    key <- paste(net, hemi, sep = "_")
    borders[[key]] <- list(
      verts   = load_npy_2d(file.path(BORDER_DIR, paste0(net, "_", hemi, "_verts.npy"))),
      weights = load_npy_2d(file.path(BORDER_DIR, paste0(net, "_", hemi, "_weights.npy"))),
      breaks  = load_npy_int(file.path(BORDER_DIR, paste0(net, "_", hemi, "_breaks.npy")))
    )
  }
}
message("Borders loaded: ", paste(names(borders), collapse = ", "))

# -- Apply gate (matches Python apply_gate + gate logic) -----------------------
gate_hemi <- function(motor, scene, msig, ssig) {
  gate <- (msig > 0.5 | ssig > 0.5) & (pmax(motor, scene, na.rm = TRUE) > pm_thresh)
  gate[is.na(gate)] <- FALSE
  motor_g <- ifelse(gate & is.finite(motor), motor, 0.0)
  scene_g <- ifelse(gate & is.finite(scene), scene, 0.0)
  list(motor = motor_g, scene = scene_g, gate = gate)
}

gL <- gate_hemi(motor_L, scene_L, msig_L, ssig_L)
gR <- gate_hemi(motor_R, scene_R, msig_R, ssig_R)
message(sprintf("Gate: LH %d / %d  RH %d / %d cortical vertices",
  sum(gL$gate), sum(is.finite(motor_L)),
  sum(gR$gate), sum(is.finite(motor_R))))

# -- Load and orient LUT -------------------------------------------------------
# Saved with row 0 = scene=vmax (vertically flipped); re-flip so that
# lut[scene_idx, motor_idx, ] is correct for scene_norm = (scene_idx-1)/255.
lut_raw <- readPNG(LUT_PATH)
lut     <- lut_raw[nrow(lut_raw):1, , , drop = FALSE]
lut_n   <- nrow(lut)  # 256

# -- Sulcal depth -> grayscale hex ---------------------------------------------
# Clip to 2nd-98th percentile across both hemispheres, normalize to [0,1],
# then rescale to display range [0.20, 0.65] (dark sulci, lighter gyri).
# Matches pycortex curvature_brightness=0.4, curvature_contrast=0.3 spirit.
sulc_all    <- c(sulc_L, sulc_R)
sulc_lo     <- quantile(sulc_all, 0.02, na.rm = TRUE)
sulc_hi     <- quantile(sulc_all, 0.98, na.rm = TRUE)
SULC_DARK   <- 0.20   # darkest (deep sulci)
SULC_LIGHT  <- 0.65   # lightest (gyral crowns)

MEDIAL_WALL_COLOR <- "#393939"

sulc_to_gray_hex <- function(sulc) {
  n       <- (pmin(pmax(sulc, sulc_lo), sulc_hi) - sulc_lo) / (sulc_hi - sulc_lo)
  g       <- SULC_DARK + n * (SULC_LIGHT - SULC_DARK)
  mwall   <- !is.finite(g)
  g[mwall] <- 0.0                  # placeholder so rgb() doesn't error
  cols    <- grDevices::rgb(g, g, g)
  cols[mwall] <- MEDIAL_WALL_COLOR
  cols
}

CORTEX_BG_COLOR <- "#808080"   # uniform gray for non-gated cortical vertices

# -- Per-vertex color: LUT for gated vertices, flat gray for the rest ----------
compute_colors <- function(motor_g, scene_g, gate, sulc) {
  # Uniform gray background; medial wall (NA sulc) gets its own color.
  cols <- ifelse(is.finite(sulc), CORTEX_BG_COLOR, MEDIAL_WALL_COLOR)
  # Overwrite gated-in vertices with the dual-band LUT color
  if (any(gate)) {
    motor_n   <- pmin(pmax(motor_g[gate], 0), vmax) / vmax
    scene_n   <- pmin(pmax(scene_g[gate], 0), vmax) / vmax
    motor_idx <- pmin(as.integer(round(motor_n * (lut_n - 1L))) + 1L, lut_n)
    scene_idx <- pmin(as.integer(round(scene_n * (lut_n - 1L))) + 1L, lut_n)
    n  <- sum(gate)
    r <- lut[cbind(scene_idx, motor_idx, rep(1L, n))]
    g <- lut[cbind(scene_idx, motor_idx, rep(2L, n))]
    b <- lut[cbind(scene_idx, motor_idx, rep(3L, n))]
    a <- lut[cbind(scene_idx, motor_idx, rep(4L, n))]
    bg <- col2rgb(CORTEX_BG_COLOR) / 255
    cols[gate] <- grDevices::rgb(r * a + bg[1] * (1 - a),
                                 g * a + bg[2] * (1 - a),
                                 b * a + bg[3] * (1 - a))
  }
  cols
}

cols_L <- compute_colors(gL$motor, gL$scene, gL$gate, sulc_L)
cols_R <- compute_colors(gR$motor, gR$scene, gR$gate, sulc_R)

# -- Surface loading and rotation ----------------------------------------------
load_surf <- function(path, reorient = FALSE) {
  s <- gifti::read_gifti(path)
  verts <- s$data[[1]]
  # Rx(-90 deg): maps MNI +Z (superior) -> rgl +Y (up). Applied to inflated
  # surfaces only; flat surfaces have their own rotation sequence.
  if (reorient) verts <- rotate_verts(verts, r1 = 0, r2 = 0, r3 = -90)
  list(
    verts = verts,
    faces = s$data[[2]] + 1L      # N x 3, 0-indexed -> 1-indexed
  )
}

# Replicates ciftiTools::rotate_surf exactly:
#   R = Rz(r1) * Ry(r2) * Rx(r3) (ZYX Euler, degrees)
#   Vertices are centered before rotation and recentered after.
rotate_verts <- function(verts, r1 = 0, r2 = 0, r3 = 0) {
  r1 <- r1 * pi / 180; r2 <- r2 * pi / 180; r3 <- r3 * pi / 180
  R <- matrix(c(
    cos(r1) * cos(r2),
    sin(r1) * cos(r2),
    -sin(r2),
    cos(r1) * sin(r2) * sin(r3) - sin(r1) * cos(r3),
    sin(r1) * sin(r2) * sin(r3) + cos(r1) * cos(r3),
    cos(r2) * sin(r3),
    cos(r1) * sin(r2) * cos(r3) + sin(r1) * sin(r3),
    sin(r1) * sin(r2) * cos(r3) - cos(r1) * sin(r3),
    cos(r2) * cos(r3)
  ), nrow = 3)
  ctr <- colMeans(verts)
  t(R %*% t(verts - matrix(ctr, nrow(verts), 3, byrow = TRUE)) +
      matrix(ctr, 3, nrow(verts)))
}

# Apply the flat-surface rotation sequence from ciftiTools so the mesh is
# oriented for a front-on view (theta=0, phi=0) in rgl.
#   LH: Ry(90) then Rx(90)
#   RH: Ry(90) then Rz(180)*Rx(90)  [rotate_surf(r1=180,r2=0,r3=90)]
rotate_flat_lh <- function(surf) {
  surf$verts <- rotate_verts(surf$verts, r1 = 0, r2 = 90, r3 = 0)
  surf$verts <- rotate_verts(surf$verts, r1 = 0, r2 =  0, r3 = 90)
  surf
}
rotate_flat_rh <- function(surf) {
  surf$verts <- rotate_verts(surf$verts, r1 =   0, r2 = 90, r3 =  0)
  surf$verts <- rotate_verts(surf$verts, r1 = 180, r2 =  0, r3 = 90)
  surf
}

# -- Border drawing ------------------------------------------------------------
# Evaluate barycentric border positions from a (already-transformed) surface
# vertex array, then draw as line segments into the current rgl scene.
# surf_verts: (N_verts x 3) matrix of rotated surface coordinates.
# hemi: "L" or "R".
# offset: numeric(3) displacement added to all border points before drawing.
# Used for flat surfaces to push borders in front of the coplanar mesh (avoids
# z-fighting).  After rotate_flat_lh/rh the surface normal points along +X,
# so a positive X offset brings the lines forward.
# max_jump: skip drawing any segment whose 3D length exceeds this threshold.
# Prevents long artifact lines across flat-surface seams.
draw_borders <- function(surf_verts, hemi, offset = c(0, 0, 0), max_jump = 20) {
  for (net in names(BORDER_COLORS)) {
    col <- BORDER_COLORS[[net]]
    bd  <- borders[[paste(net, hemi, sep = "_")]]
    if (is.null(bd) || nrow(bd$verts) == 0L) next

    v0 <- bd$verts[, 1] + 1L
    v1 <- bd$verts[, 2] + 1L
    v2 <- bd$verts[, 3] + 1L
    pts <- bd$weights[, 1] * surf_verts[v0, ] +
           bd$weights[, 2] * surf_verts[v1, ] +
           bd$weights[, 3] * surf_verts[v2, ]
    pts <- pts + matrix(offset, nrow = nrow(pts), ncol = 3L, byrow = TRUE)

    part_starts <- c(bd$breaks + 1L, nrow(pts) + 1L)
    for (i in seq_len(length(part_starts) - 1L)) {
      seg <- pts[seq(part_starts[i], part_starts[i + 1L] - 1L), , drop = FALSE]
      if (nrow(seg) < 2L) next
      # Detect seam jumps: split the part at large inter-point distances.
      diffs  <- diff(seg)
      dists  <- sqrt(rowSums(diffs^2))
      breaks <- c(0L, which(dists > max_jump), nrow(seg))
      for (j in seq_len(length(breaks) - 1L)) {
        sub <- seg[seq(breaks[j] + 1L, breaks[j + 1L]), , drop = FALSE]
        if (nrow(sub) < 10L) next   # drop dangling seam-edge stubs
        rgl::lines3d(sub[, 1], sub[, 2], sub[, 3],
                     col = col, lwd = BORDER_LWD, lit = FALSE)
      }
    }
  }
}

# -- rgl rendering -------------------------------------------------------------
render_and_save <- function(surf, vert_colors, theta, phi, zoom,
                             width, height, out_file, bg = "white",
                             both_sides = FALSE, ambient = "grey20",
                             hemi = NULL, border_offset = c(0, 0, 0),
                             max_jump = 20, clip_background = FALSE) {
  mesh <- rgl::tmesh3d(
    vertices    = t(surf$verts),
    indices     = t(surf$faces),
    homogeneous = FALSE
  )
  rgl::open3d(windowRect = c(0L, 0L, width, height), silent = TRUE)
  on.exit(rgl::close3d(), add = TRUE)
  rgl::bg3d(color = bg)
  back_arg <- if (both_sides) "filled" else "culled"
  rgl::shade3d(mesh, col = vert_colors, specular = "black", lit = TRUE,
               ambient = ambient, back = back_arg)
  if (!is.null(hemi)) draw_borders(surf$verts, hemi, offset = border_offset, max_jump = max_jump)
  rgl::view3d(theta = theta, phi = phi, fov = 0, zoom = zoom)
  rgl::snapshot3d(out_file, webshot = FALSE)
  if (clip_background) clip_exterior_pixels(out_file)
  trim_png(out_file, pad = 4L)
  message("  -> ", out_file)
}

# Flood-fill the white background from each image corner (PIL, fast C impl).
# Any non-white pixel reached = border artifact outside brain; set to white.
clip_exterior_pixels <- function(path) {
  py_code <- sprintf(
"from PIL import Image, ImageDraw
import numpy as np
img = Image.open('%s').convert('RGB')
arr = np.array(img)
h, w = arr.shape[:2]
sentinel = (254, 1, 2)
for corner in [(0,0),(0,w-1),(h-1,0),(h-1,w-1)]:
    tmp = img.copy()
    ImageDraw.floodfill(tmp, (corner[1], corner[0]), sentinel, thresh=15)
    t = np.array(tmp)
    bg = (t[:,:,0]==254) & (t[:,:,1]==1) & (t[:,:,2]==2)
    white = (arr[:,:,0]>245) & (arr[:,:,1]>245) & (arr[:,:,2]>245)
    arr[bg & ~white] = [255, 255, 255]
    img = Image.fromarray(arr)
img.save('%s')
", path, path)
  system2("/opt/miniconda3/bin/conda",
          c("run", "-n", "mario", "python", "-c", shQuote(py_code)),
          stdout = FALSE, stderr = FALSE)
}

flip_png_horizontal <- function(path) {
  img <- png::readPNG(path)
  png::writePNG(img[, ncol(img):1, , drop = FALSE], path)
}

# Crop white border from a PNG saved by rgl (RGB, white background).
# Keeps `pad` pixels of padding on each side.
trim_png <- function(path, pad = 4L) {
  img  <- png::readPNG(path)
  # Non-white mask: any channel < 1 (white = 1,1,1 in [0,1] float)
  mask <- img[, , 1] < 1 | img[, , 2] < 1 | img[, , 3] < 1
  if (!any(mask)) return(invisible(NULL))
  rows <- which(apply(mask, 1, any))
  cols <- which(apply(mask, 2, any))
  r1 <- max(1L,        min(rows) - pad)
  r2 <- min(nrow(img), max(rows) + pad)
  c1 <- max(1L,        min(cols) - pad)
  c2 <- min(ncol(img), max(cols) + pad)
  png::writePNG(img[r1:r2, c1:c2, , drop = FALSE], path)
}

# Camera conventions for HCP fs_LR (MNI: +X=right, +Y=ant, +Z=sup).
# rgl view3d: theta rotates around vertical (Y) axis; phi tilts elevation.
# Default view is along -Z (camera at +Z).
#
# Inflated surfaces -- LH sits at negative X, RH at positive X:
#   LH lateral:  theta =  90  (camera from -X side)
#   LH medial:   theta = -90  (camera from +X side)
#   RH lateral:  theta = -90
#   RH medial:   theta =  90
#
# Flat surfaces -- after the ciftiTools rotation sequence the mesh is in the
# YZ plane (x = constant), so a front view (theta=0, phi=0) shows the 2D map.

INF_ZOOM  <- 0.82
FLAT_ZOOM <- 0.88

if (opt$views %in% c("all", "inflated")) {
  message("\nRendering inflated surfaces...")
  surf_L_inf <- load_surf(SURF_L_INF, reorient = TRUE)
  surf_R_inf <- load_surf(SURF_R_INF, reorient = TRUE)
  render_and_save(surf_L_inf, cols_L, theta = -90, phi = 0,
    zoom = INF_ZOOM, width = render_width, height = render_height,
    out_file = file.path(OUT_DIR, "dual_band_inflated_lh_lateral.png"), hemi = "L")
  render_and_save(surf_L_inf, cols_L, theta =  90, phi = 0,
    zoom = INF_ZOOM, width = render_width, height = render_height,
    out_file = file.path(OUT_DIR, "dual_band_inflated_lh_medial.png"),  hemi = "L")
  render_and_save(surf_R_inf, cols_R, theta =  90, phi = 0,
    zoom = INF_ZOOM, width = render_width, height = render_height,
    out_file = file.path(OUT_DIR, "dual_band_inflated_rh_lateral.png"), hemi = "R")
  render_and_save(surf_R_inf, cols_R, theta = -90, phi = 0,
    zoom = INF_ZOOM, width = render_width, height = render_height,
    out_file = file.path(OUT_DIR, "dual_band_inflated_rh_medial.png"),  hemi = "R")
}

if (opt$views %in% c("all", "flatmap")) {
  message("\nRendering flat surfaces...")
  surf_L_flat <- rotate_flat_lh(load_surf(SURF_L_FLAT))
  surf_R_flat <- rotate_flat_rh(load_surf(SURF_R_FLAT))
  FLAT_BORDER_OFFSET <- c(3, 0, 0)   # push borders in front of coplanar mesh
  render_and_save(surf_L_flat, cols_L, theta = 90, phi = 0,
    zoom = FLAT_ZOOM, width = render_width, height = render_height,
    out_file = file.path(OUT_DIR, "dual_band_flatmap_lh.png"),
    both_sides = TRUE, ambient = "grey10", hemi = "L",
    border_offset = FLAT_BORDER_OFFSET, max_jump = 8, clip_background = TRUE)
  rh_flat_path <- file.path(OUT_DIR, "dual_band_flatmap_rh.png")
  render_and_save(surf_R_flat, cols_R, theta = 90, phi = 0,
    zoom = FLAT_ZOOM, width = render_width, height = render_height,
    out_file = rh_flat_path, both_sides = TRUE, ambient = "grey10", hemi = "R",
    border_offset = FLAT_BORDER_OFFSET, max_jump = 8, clip_background = TRUE)
  flip_png_horizontal(rh_flat_path)
}

# -- Legend --------------------------------------------------------------------
message("\nGenerating legend...")
legend_path <- file.path(OUT_DIR, "dual_band_legend.png")
legend_script <- file.path(REPO, "scripts/python/plot_vp_legend.py")
ret <- system2(
  "/opt/miniconda3/bin/conda",
  c("run", "-n", "mario", "python", legend_script,
    "--out",  legend_path,
    "--vmax", as.character(vmax),
    "--dpi",  as.character(opt$dpi)),
  stdout = TRUE, stderr = TRUE
)
if (sum(grepl("Legend saved", ret)) < 1L) {
  message("Python legend failed:\n", paste(ret, collapse = "\n"))
} else {
  message("  -> ", legend_path)
}
message("\nDone.")
