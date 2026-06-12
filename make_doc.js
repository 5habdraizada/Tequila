const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak, TableOfContents,
  LevelFormat, ExternalHyperlink
} = require("docx");
const fs = require("fs");

// ── Colour palette ──────────────────────────────────────────────────────────
const BLUE       = "1F4E79";
const BLUE_LIGHT = "2E75B6";
const BLUE_PALE  = "D6E4F0";
const BLUE_MID   = "BDD7EE";
const GREY_LIGHT = "F2F2F2";
const GREY_MID   = "D9D9D9";
const WHITE      = "FFFFFF";

// ── Borders ─────────────────────────────────────────────────────────────────
const cellBorder = (color = "CCCCCC") => ({
  top:    { style: BorderStyle.SINGLE, size: 1, color },
  bottom: { style: BorderStyle.SINGLE, size: 1, color },
  left:   { style: BorderStyle.SINGLE, size: 1, color },
  right:  { style: BorderStyle.SINGLE, size: 1, color },
});
const noBorder = () => ({
  top:    { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  left:   { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  right:  { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
});

// ── Helpers ──────────────────────────────────────────────────────────────────
const h1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  children: [new TextRun({ text, font: "Arial", bold: true })],
  spacing: { before: 360, after: 120 },
});

const h2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  children: [new TextRun({ text, font: "Arial", bold: true })],
  spacing: { before: 280, after: 80 },
});

const h3 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_3,
  children: [new TextRun({ text, font: "Arial", bold: true })],
  spacing: { before: 200, after: 60 },
});

const p = (...runs) => new Paragraph({
  children: runs.map(r =>
    typeof r === "string"
      ? new TextRun({ text: r, font: "Arial", size: 22 })
      : r
  ),
  spacing: { before: 60, after: 60 },
});

const pIndent = (...runs) => new Paragraph({
  children: runs.map(r =>
    typeof r === "string"
      ? new TextRun({ text: r, font: "Arial", size: 22 })
      : r
  ),
  indent: { left: 720 },
  spacing: { before: 40, after: 40 },
});

const bold = (text) => new TextRun({ text, font: "Arial", size: 22, bold: true });
const italic = (text) => new TextRun({ text, font: "Arial", size: 22, italics: true });
const code = (text) => new TextRun({ text, font: "Courier New", size: 20, color: "C0392B" });
const normal = (text) => new TextRun({ text, font: "Arial", size: 22 });

const bullet = (text, level = 0) => new Paragraph({
  numbering: { reference: "bullets", level },
  children: [new TextRun({ text, font: "Arial", size: 22 })],
  spacing: { before: 40, after: 40 },
});

const bulletMixed = (runs, level = 0) => new Paragraph({
  numbering: { reference: "bullets", level },
  children: runs.map(r => typeof r === "string" ? normal(r) : r),
  spacing: { before: 40, after: 40 },
});

const numbered = (text, level = 0) => new Paragraph({
  numbering: { reference: "numbered", level },
  children: [new TextRun({ text, font: "Arial", size: 22 })],
  spacing: { before: 40, after: 40 },
});

const numberedMixed = (runs, level = 0) => new Paragraph({
  numbering: { reference: "numbered", level },
  children: runs.map(r => typeof r === "string" ? normal(r) : r),
  spacing: { before: 40, after: 40 },
});

const spacer = (pt = 120) => new Paragraph({
  children: [],
  spacing: { before: 0, after: pt },
});

const codeBlock = (lines) => {
  return lines.map(line => new Paragraph({
    children: [new TextRun({ text: line, font: "Courier New", size: 18, color: "EEEEEE" })],
    shading: { fill: "1E1E1E", type: ShadingType.CLEAR },
    spacing: { before: 0, after: 0 },
    indent: { left: 360, right: 360 },
  }));
};

// Simple two-column header row for tables
const tableHeaderRow = (cells, widths) => new TableRow({
  tableHeader: true,
  children: cells.map((text, i) => new TableCell({
    borders: cellBorder(BLUE_LIGHT),
    width: { size: widths[i], type: WidthType.DXA },
    shading: { fill: BLUE_LIGHT, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      children: [new TextRun({ text, font: "Arial", size: 20, bold: true, color: WHITE })],
      alignment: AlignmentType.LEFT,
    })],
  })),
});

const tableDataRow = (cells, widths, shade = false) => new TableRow({
  children: cells.map((cell, i) => new TableCell({
    borders: cellBorder("CCCCCC"),
    width: { size: widths[i], type: WidthType.DXA },
    shading: { fill: shade ? GREY_LIGHT : WHITE, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: Array.isArray(cell)
      ? [new Paragraph({ children: cell.map(r => typeof r === "string" ? normal(r) : r) })]
      : [new Paragraph({ children: [new TextRun({ text: cell, font: "Arial", size: 20 })] })],
  })),
});

const makeTable = (headers, rows, widths, totalWidth) => new Table({
  width: { size: totalWidth, type: WidthType.DXA },
  columnWidths: widths,
  rows: [
    tableHeaderRow(headers, widths),
    ...rows.map((row, i) => tableDataRow(row, widths, i % 2 === 0)),
  ],
});

// ── Section divider ──────────────────────────────────────────────────────────
const divider = () => new Paragraph({
  children: [],
  border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: BLUE_LIGHT, space: 1 } },
  spacing: { before: 160, after: 160 },
});

// ── Page break ───────────────────────────────────────────────────────────────
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

// ── DOCUMENT ─────────────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
          { level: 1, format: LevelFormat.BULLET, text: "\u25E6", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
        ],
      },
      {
        reference: "numbered",
        levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
          { level: 1, format: LevelFormat.LOWER_LETTER, text: "%2.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
        ],
      },
    ],
  },

  styles: {
    default: {
      document: { run: { font: "Arial", size: 22 } },
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: BLUE },
        paragraph: { spacing: { before: 360, after: 120 }, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: BLUE_LIGHT, space: 4 } } },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: BLUE_LIGHT },
        paragraph: { spacing: { before: 280, after: 80 }, outlineLevel: 1 },
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "2F5597" },
        paragraph: { spacing: { before: 200, after: 60 }, outlineLevel: 2 },
      },
    ],
  },

  sections: [
    // ── SECTION 1 — Title page ────────────────────────────────────────────────
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      children: [
        spacer(2880),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "TEQUILA", font: "Arial", size: 72, bold: true, color: BLUE })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Live 3D Mapping + Navigation Mesh", font: "Arial", size: 36, color: BLUE_LIGHT })],
          spacing: { before: 120, after: 240 },
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          border: { top: { style: BorderStyle.SINGLE, size: 4, color: BLUE_LIGHT, space: 8 },
                    bottom: { style: BorderStyle.SINGLE, size: 4, color: BLUE_LIGHT, space: 8 } },
          children: [new TextRun({ text: "Technical Architecture & System Documentation", font: "Arial", size: 26, italics: true, color: "444444" })],
          spacing: { before: 120, after: 120 },
        }),
        spacer(480),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "A real-time indoor mapping and frontier exploration system", font: "Arial", size: 24, color: "555555" })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "combining monocular metric depth estimation with navigation mesh generation", font: "Arial", size: 24, color: "555555" })],
          spacing: { before: 0, after: 480 },
        }),
        spacer(2400),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Version 1.0  |  2026", font: "Arial", size: 22, color: "888888" })],
        }),
        pageBreak(),

        // ── Table of Contents ─────────────────────────────────────────────────
        new Paragraph({
          children: [new TextRun({ text: "Contents", font: "Arial", size: 36, bold: true, color: BLUE })],
          spacing: { before: 0, after: 240 },
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: BLUE_LIGHT, space: 4 } },
        }),
        new TableOfContents("", {
          hyperlink: true,
          headingStyleRange: "1-3",
          stylesWithLevels: [],
        }),
        pageBreak(),

        // ════════════════════════════════════════════════════════════════════
        // 1. OVERVIEW
        // ════════════════════════════════════════════════════════════════════
        h1("1.  Overview"),
        p("TEQUILA is a real-time 3D mapping and navigation system designed for indoor mobile robots. It takes a live video stream from a single camera, builds a dense coloured point-cloud map of the environment, generates a navigation mesh on the detected floor, and plans a path to the furthest reachable point — all simultaneously and in real time."),
        spacer(80),
        p("The system is built around three key ideas:"),
        bullet("No special hardware required — a standard webcam and a consumer GPU (or even a CPU) are sufficient."),
        bullet("Metric depth without a depth camera — Depth Anything V2, a transformer-based neural network, estimates real distances in metres from a single RGB image."),
        bullet("Frontier exploration by design — the navigation mesh always paths to the furthest reachable waypoint, driving the robot towards unexplored space."),
        spacer(120),

        h2("1.1  What the system produces"),
        p("Every few seconds TEQUILA produces three outputs, all visible live in a browser-based 3D viewer:"),
        spacer(60),
        makeTable(
          ["Output", "Description", "Update rate"],
          [
            ["Coloured point cloud", "A dense world-space 3D map built by fusing depth from every processed frame", "Every frame"],
            ["Navigation mesh", "Floor plane, free waypoint nodes, obstacles, and passable edges", "Every 6 s (configurable)"],
            ["A* path", "Shortest path from the robot to the furthest reachable waypoint", "Every 6 s (configurable)"],
          ],
          [3500, 4360, 1500],
          9360
        ),
        spacer(120),

        h2("1.2  Thread architecture"),
        p("TEQUILA runs four concurrent threads so that capture, inference, navmesh computation, and display never block each other. Each thread communicates with the next through a single-slot queue (maxsize = 1). This means every consumer always sees the most recent data — stale items are automatically discarded when a newer item arrives."),
        spacer(80),
        makeTable(
          ["Thread", "Responsibility", "Produces"],
          [
            ["CaptureThread", "Reads frames from webcam or video file at a configurable rate", "Raw BGR frames"],
            ["InferenceThread", "Runs the depth model, removes flying pixels, back-projects to 3D", "Nav point cloud + coloured map cloud"],
            ["NavmeshThread", "Aligns frames, accumulates world map, computes navmesh and A* path", "Navmesh overlay + trajectory"],
            ["Main thread (Viewer)", "Polls the output queues at ~20 Hz and updates the viser 3D scene", "Browser visualisation"],
          ],
          [2200, 4460, 2700],
          9360
        ),
        spacer(160),

        pageBreak(),

        // ════════════════════════════════════════════════════════════════════
        // 2. DEPTH INFERENCE
        // ════════════════════════════════════════════════════════════════════
        h1("2.  Depth Inference"),
        p("Depth inference is the foundation of everything else in the pipeline. Every frame must be converted from a flat 2D image into a 3D point cloud before it can be aligned, accumulated, or used for navmesh generation."),

        h2("2.1  The depth model"),
        p("TEQUILA uses Depth Anything V2, a vision transformer trained to predict per-pixel depth from a single RGB image. The default model is the Metric Indoor Small variant (24.8 million parameters, ~100 MB). It outputs depth values in real metres with no per-frame normalisation — every pixel value directly represents a physical distance from the camera lens."),
        spacer(80),
        p("This is critical. Many monocular depth models are relative — they normalise each frame independently so that the nearest point in that frame maps to 0 and the furthest maps to some maximum value. This means the same physical distance produces a different depth value in every frame. When those frames are accumulated into a world map, the scale differences cause the cloud to fan and stretch badly. The metric model eliminates this problem entirely."),
        spacer(80),
        makeTable(
          ["Variant", "Parameters", "RMSE", "δ₁ Accuracy", "Download size"],
          [
            ["Small (default)", "24.8 M", "0.261 m", "96.1 %", "~100 MB"],
            ["Large", "335 M", "0.206 m", "98.4 %", "~1.3 GB"],
          ],
          [2000, 1800, 1600, 2000, 1960],
          9360
        ),
        spacer(60),
        p("The 2.3 percentage-point accuracy difference between Small and Large rarely matters in practice for detecting chairs and walls at typical robot operating distances of 1–4 metres."),

        h2("2.2  Why flying pixels appear — and how they are removed"),
        p("At the boundary between a near object and a far background, depth models interpolate across the edge and produce intermediate depth values on the boundary pixels. When back-projected to 3D, these pixels appear neither at the near surface nor at the far surface — they float in between, forming long rays or spikes extending behind every object edge."),
        spacer(80),
        p("TEQUILA removes these flying pixels using morphological erosion. For every pixel, the local depth minimum is computed across an 11 x 11 pixel neighbourhood using an erosion kernel. Any pixel whose depth exceeds that local minimum by more than 20 % is classified as a flying pixel and masked out:"),
        spacer(80),
        ...codeBlock([
          "  local_min = erode(depth_map, 11x11 kernel)",
          "  valid     = depth <= local_min * 1.20",
          "  depth     = where(valid, depth, 0.0)     # 0 = masked out",
        ]),
        spacer(80),
        p("Masked pixels are set to zero. The zero-depth filter in the back-projection step then excludes them: after flipping Z to the navigation convention, masked pixels have z = 0 and are removed by the check z < 0."),

        h2("2.3  Back-projection to 3D"),
        p("Valid depth pixels are lifted to 3D using the standard pinhole camera model. The principal point is assumed to be at the image centre and the focal length is derived from the configured horizontal field of view:"),
        spacer(80),
        ...codeBlock([
          "  focal = width / (2 * tan(FOV_horizontal / 2))",
          "",
          "  x =  (pixel_x - cx) * depth / focal",
          "  y = -(pixel_y - cy) * depth / focal   # flip Y  (image Y-down -> nav Y-up)",
          "  z = -depth                             # flip Z  (into-scene -> toward-viewer)",
        ]),
        spacer(80),
        p("The Y and Z flips convert from standard OpenCV camera coordinates (Y pointing down, Z pointing into the scene) to the navigation coordinate convention used throughout the rest of the system (Y pointing up, Z pointing toward the viewer)."),
        spacer(80),
        p("Two separate point clouds are produced from each frame:"),
        bulletMixed([bold("nav_pts"), " — coarse (3x voxel grid), position-only. Used for floor plane fitting and navmesh generation."]),
        bulletMixed([bold("map_pts"), " — fine (1x voxel grid), with colours from the original image. Used for the accumulated display map."]),
        spacer(160),

        pageBreak(),

        // ════════════════════════════════════════════════════════════════════
        // 3. FRAME ALIGNMENT
        // ════════════════════════════════════════════════════════════════════
        h1("3.  Frame Alignment"),
        p("Building a consistent world map requires every frame to be expressed in the same coordinate system. Frame alignment estimates the rigid transform (rotation + translation) between consecutive frames, then composes those transforms into a single cumulative world transform T_cum. Each frame's points are then expressed in world space as:"),
        spacer(60),
        ...codeBlock([
          "  world_pts = (R_w @ cam_pts.T).T + t_w",
          "  where  R_w, t_w = T_cum[:3,:3],  T_cum[:3,3]",
        ]),
        spacer(80),
        p("TEQUILA tries two alignment methods in order each frame: ORB + PnP (primary) and ICP (fallback)."),

        h2("3.1  ORB + PnP Visual Odometry (primary)"),
        p("Visual odometry estimates camera motion from image features. It is fast and accurate on textured environments where enough keypoints can be matched between frames."),
        spacer(80),

        h3("Step 1 — ORB keypoint detection"),
        p("The Oriented FAST and Rotated BRIEF (ORB) detector finds up to 2,000 keypoints in each frame. ORB is chosen because it is rotation-invariant, scale-invariant, and fast enough to run on CPU. Each keypoint is described by a 256-bit binary descriptor."),

        h3("Step 2 — Descriptor matching and Lowe's ratio test"),
        p("Descriptors from the previous frame are matched to descriptors in the current frame using a brute-force Hamming distance matcher with k=2 nearest neighbours. A match is kept only if its distance is less than 75 % of the distance to the second-best match:"),
        spacer(60),
        ...codeBlock([
          "  good = [m for m, n in matches if m.distance < 0.75 * n.distance]",
        ]),
        spacer(60),
        p("This ratio test (Lowe 2004) rejects ambiguous matches — those where the best and second-best candidate are similar distances apart, meaning the true correspondence is uncertain."),

        h3("Step 3 — 3D back-projection of matched keypoints"),
        p("Each matched keypoint from the previous frame is looked up in the stored depth map and back-projected to 3D. This uses standard OpenCV camera coordinates (Y-down, Z-into-scene) so that the camera matrix K applies directly without any axis flipping:"),
        spacer(60),
        ...codeBlock([
          "  d = depth_prev[v1, u1]          # depth at matched pixel",
          "  x = (u1 - cx) * d / focal",
          "  y = (v1 - cy) * d / focal       # no flip here",
          "  z = d",
        ]),
        spacer(60),
        p("Keypoints with missing depth (d < 0.05 m) or saturated depth (d >= MAX_DEPTH * 0.99) are discarded."),

        h3("Step 4 — solvePnPRansac"),
        p("The set of 3D points (from the previous frame) and their corresponding 2D locations (in the current frame) is passed to OpenCV's solvePnPRansac. This fits the camera pose that minimises reprojection error — the distance between where each 3D point projects onto the current image and where the matched keypoint actually is. RANSAC makes the fit robust to incorrect matches."),
        spacer(60),
        p("The output is a rotation vector rvec and translation vector tvec in standard camera coordinates."),

        h3("Step 5 — Coordinate convention conversion"),
        p("The result from solvePnPRansac is in standard camera coordinates. It must be converted to the navigation convention (Y-up, Z-toward-viewer) using the flip matrix F = diag(1, -1, -1):"),
        spacer(60),
        ...codeBlock([
          "  F = diag([1, -1, -1])",
          "  R_nav = F @ R_std @ F",
          "  t_nav = F @ t_std",
        ]),
        spacer(60),
        p("The result is then inverted so it maps current-frame points into previous-frame coordinates, matching the convention expected by the accumulation step:"),
        spacer(60),
        ...codeBlock([
          "  R_rel = R_curr_from_prev.T",
          "  t_rel = -(R_curr_from_prev.T @ t_curr_from_prev)",
        ]),

        h3("Sanity checks"),
        p("A result is accepted only if it passes all three checks:"),
        bullet("At least 12 PnP inliers (matches that agree with the estimated pose)"),
        bullet("Translation between 0.03 m and 2.0 m — frames with less than 3 cm of motion are treated as duplicate views and skipped (the camera has not moved enough to add new information); translations above 2 m indicate a wild alignment failure"),
        bullet("Rotation less than 45 degrees"),

        h2("3.2  ICP frame alignment (fallback)"),
        p("If ORB+PnP fails — most often because the environment has too few texture features (plain white walls, low light, motion blur) — ICP is tried on the raw nav point clouds from the current and previous frames."),
        spacer(80),
        p("ICP finds the rigid transform that minimises the total distance between each source point and its nearest neighbour in the target cloud. Each iteration:"),
        numbered("Find the nearest neighbour in the target cloud for every source point using a KDTree"),
        numbered("Keep only pairs within 1.0 m of each other (inliers)"),
        numbered("Compute the optimal rigid transform via the Umeyama SVD method:"),
        spacer(40),
        ...codeBlock([
          "  H = (source_inliers - src_centroid).T @ (target_inliers - tgt_centroid)",
          "  U, _, Vt = SVD(H)",
          "  R = Vt.T @ U.T",
          "  t = tgt_centroid - R @ src_centroid",
        ]),
        spacer(60),
        numbered("Apply the transform to the source cloud and repeat for up to 50 iterations or until convergence (change in mean inlier distance < 0.0001 m)"),
        spacer(80),
        p("ICP is accepted if: fitness (fraction of source points within 1 m of a match) >= 0.25, translation < 2 m, rotation < 45 degrees. If both VO and ICP fail, the frame is skipped entirely and the world transform is not updated."),
        spacer(160),

        pageBreak(),

        // ════════════════════════════════════════════════════════════════════
        // 4. MAP ACCUMULATION
        // ════════════════════════════════════════════════════════════════════
        h1("4.  Map Accumulation"),
        p("Once a frame is aligned, its points are transformed into world space using the cumulative transform and merged into the growing world map. Two separate clouds are maintained for different purposes."),

        h2("4.1  The two accumulated clouds"),
        makeTable(
          ["Cloud", "Resolution", "Contents", "Purpose"],
          [
            ["Coarse nav cloud", "3x voxel (6 cm)", "Positions only", "Floor plane fitting, navmesh RANSAC"],
            ["Fine coloured map", "1x voxel (2 cm)", "Positions + RGB colours", "Browser display, trajectory context"],
          ],
          [2000, 2000, 2160, 3200],
          9360
        ),
        spacer(80),
        p("Only points within MAP_MAX_DEPTH (default 4 m) are added to the fine coloured map. Distant points amplify any alignment error — a 1 degree rotation error at 4 m produces an 7 cm positional error; at 8 m it doubles to 14 cm. Capping the depth limits the visible effect of drift."),

        h2("4.2  Memory management"),
        p("Both clouds are periodically voxel-downsampled to prevent unbounded memory growth. Voxel downsampling divides space into a regular grid and keeps only one point per cell. Additional SOR (Statistical Outlier Removal) passes remove isolated noise points."),
        spacer(80),
        p("Both clouds are hard-capped at 500,000 points. When the cap is reached the oldest half is discarded, keeping the most recently observed geometry."),

        h2("4.3  Duplicate-view suppression"),
        p("If ORB+PnP succeeds but reports a translation of less than 0.03 m, the camera has barely moved. The alignment state is updated (so the next frame aligns correctly) but no points are added to the map. This prevents the same view being accumulated multiple times, which would add density in one place without adding new information."),
        spacer(160),

        pageBreak(),

        // ════════════════════════════════════════════════════════════════════
        // 5. FLOOR DETECTION
        // ════════════════════════════════════════════════════════════════════
        h1("5.  Floor Detection"),
        p("The navmesh pipeline begins with floor detection. The system must identify which points in the accumulated cloud belong to the floor and which are obstacles. This is done by fitting a plane to the floor surface."),
        p("TEQUILA uses the Ground Principal Plane (GPP) algorithm as its primary method, with classic RANSAC as a fallback."),

        h2("5.1  Ground Principal Plane (GPP) algorithm"),
        p("GPP divides the point cloud into angular sectors around the vertical axis and fits a plane independently to each sector. The largest group of sectors that agree on the floor orientation defines the consensus floor plane. This is more robust than single-plane RANSAC because it handles partial occlusion and uneven point density across the room."),
        spacer(80),

        h3("Step 1 — Sector partitioning"),
        p("The point cloud is divided into 8 sectors of 45 degrees each, centred on the camera. Each sector is a slice of the horizontal angle around the up-axis. This partitioning ensures the algorithm sees floor samples from multiple directions around the robot, not just the densest part of the cloud."),

        h3("Step 2 — Per-sector PCA plane fitting"),
        p("For each sector with at least 10 points, a plane is fitted using Principal Component Analysis. The three columns of the data matrix form three principal directions; the eigenvector corresponding to the smallest eigenvalue (smallest singular value) is the normal to the flattest plane through the sector points:"),
        spacer(60),
        ...codeBlock([
          "  centroid = sector_points.mean(axis=0)",
          "  _, _, Vt = SVD(sector_points - centroid,  full_matrices=False)",
          "  normal   = Vt[-1]      # row with smallest singular value",
        ]),
        spacer(60),
        p("Sectors whose fitted plane normal is more than 30 degrees from the up-axis are discarded as too tilted to be a floor."),

        h3("Step 3 — Consensus voting"),
        p("The surviving sector normals are compared pairwise. The largest group whose normals all agree within 15 degrees of each other forms the consensus. Agreement is measured as the dot product between unit normals:"),
        spacer(60),
        ...codeBlock([
          "  agree = (normals @ normals[i]) >= cos(15 degrees)",
        ]),
        spacer(60),
        p("This consensus step makes the algorithm resilient to sectors that accidentally fit a tilted surface (ramp, foot of a sofa) — those outlier normals will not agree with the majority and are excluded."),

        h3("Step 4 — Least-squares refinement"),
        p("The consensus normals are averaged to get an initial floor normal. All points in the full cloud within 20 cm of the plane defined by this normal are collected as rough inliers. A final least-squares PCA refinement is run on these inliers to produce the definitive floor plane."),

        h2("5.2  Floor validity check"),
        p("After GPP produces a candidate floor, it is validated against the camera position: the mean Y coordinate of the floor inliers must be at least 5 cm below the camera along the up-axis. If GPP returns a surface that is above the camera — a table top, ceiling, or wall accidentally promoted due to VO drift — it is rejected immediately and RANSAC is retried."),
        spacer(80),
        p("This check is critical because VO drift can rotate the accumulated cloud slightly over time. A wall that is physically vertical may appear 20 to 30 degrees from vertical in the accumulated frame, making its plane pass the tilt threshold. The camera-below constraint eliminates these false positives."),

        h2("5.3  RANSAC fallback"),
        p("If GPP cannot find enough sector agreement (e.g. very sparse point cloud, strong drift), classic RANSAC runs instead. It randomly samples triplets of points, fits a plane, and counts inliers within 20 cm. Only candidate planes whose inlier centroid is below the camera are considered. The plane with the most inliers is refined with least-squares."),
        spacer(160),

        pageBreak(),

        // ════════════════════════════════════════════════════════════════════
        // 6. NAVIGATION MESH
        // ════════════════════════════════════════════════════════════════════
        h1("6.  Navigation Mesh Generation"),
        p("Once the floor plane is detected, the navmesh pipeline places waypoint nodes on the floor, removes nodes blocked by obstacles, connects free nodes with edges, and plans a path through the result."),

        h2("6.1  Node grid placement"),
        p("Floor inlier points are projected to 2D floor coordinates using two orthonormal in-plane axes U and V, constructed from the floor normal:"),
        spacer(60),
        ...codeBlock([
          "  U = cross(normal, up) / |cross(normal, up)|",
          "  V = cross(normal, U)  / |cross(normal, U)|",
          "",
          "  u_coords = floor_pts @ U",
          "  v_coords = floor_pts @ V",
        ]),
        spacer(60),
        p("A regular grid at 15 cm spacing is laid across the extent of the projected floor. For each grid cell, the system checks whether a floor point exists within 22.5 cm (1.5 times the node spacing). If yes, a node is placed at that grid position."),
        spacer(80),
        p("Critically, nodes are projected directly onto the mathematical plane equation rather than being snapped to the nearest floor point's 3D position. Since U, V, and normal form an orthonormal basis, and any point P on the plane satisfies normal dot P = -d, the 3D position of a node at grid coordinates (u, v) is exactly:"),
        spacer(60),
        ...codeBlock([
          "  P = u*U + v*V + (-d)*normal",
        ]),
        spacer(60),
        p("This ensures all nodes sit at exactly the same height regardless of how much the accumulated floor points are spread vertically by depth noise or VO drift. Using the nearest floor point position instead would cause some nodes to appear below or above the floor surface."),
        spacer(60),
        p("Each node is lifted NODE_ABOVE_FLOOR (4 cm) along the floor normal to avoid z-fighting with the floor mesh in the viewer."),

        h2("6.2  Obstacle detection"),
        p("Non-floor points are candidate obstacles. They are filtered by height above the floor using the absolute world Y coordinate (not the tilted plane distance). The tilted plane distance is corrupted by VO rotational drift — a point on the ceiling could appear only 0.3 m above a tilted detected plane. Using the world Y axis instead gives a stable physical height measurement:"),
        spacer(60),
        ...codeBlock([
          "  floor_y  = floor_pts[:, up_idx].mean()",
          "  height   = obs_pts[:, up_idx] - floor_y",
          "  obstacles = obs_pts[(height > 0.18) & (height < 0.80)]",
        ]),
        spacer(60),
        p("Points below 18 cm are ignored — this band absorbs depth noise and VO drift that would otherwise cause floor-level points to be falsely classified as obstacles. Points above 80 cm are also ignored (ceiling, tall furniture) since the robot can pass under them."),
        spacer(80),
        p("The remaining obstacle points are voxel-downsampled at 5 cm and, for small clouds, Statistical Outlier Removal is applied to remove isolated noise points. Two clouds are produced: a tight-SOR version for display (orange dots in the viewer) and a loose-SOR version for collision checks."),

        h2("6.3  Node filtering"),
        p("Any navmesh node within OBS_CLEARANCE_R (40 cm) of an obstacle is marked as blocked. The clearance radius should be at least half the robot body width. Blocked nodes are shown in red in the viewer."),

        h2("6.4  Edge building and line-of-sight check"),
        p("Pairs of free nodes within EDGE_MAX_DIST (30 cm, which is 2x the node spacing) are candidate edges. For each candidate pair, 20 equally-spaced points are sampled along the connecting line segment. If any sample point is within 40 cm of an obstacle, the edge is rejected:"),
        spacer(60),
        ...codeBlock([
          "  for i, j in candidate_pairs:",
          "      samples = linspace(node[i], node[j], 20 points)",
          "      if min(distance(samples, obstacles)) >= 0.40:",
          "          add_edge(i, j)",
        ]),
        spacer(60),
        p("This line-of-sight check ensures the robot never tries to drive through a gap it physically cannot fit through, even if both endpoints of an edge are technically free."),

        h2("6.5  A* path planning — frontier exploration"),
        p("A* finds the shortest path through the free-node graph from the start node to the goal node. The heuristic is Euclidean distance to the goal, making A* optimal for this metric graph."),
        spacer(80),
        p("The goal is chosen using a frontier exploration strategy: nodes are sorted by distance from the camera (descending) and A* is tried in order from farthest to nearest until a connected path is found. This means the robot always heads for the most distant area that the map shows as reachable — naturally driving exploration outward from the robot's current position into unknown space."),
        spacer(80),
        p("This is a greedy frontier strategy (always go to the most distant reachable point). A complete frontier exploration system would additionally track which areas have already been visited and exclude them from future goals, but this simpler approach works well for initial mapping passes."),
        spacer(160),

        pageBreak(),

        // ════════════════════════════════════════════════════════════════════
        // 7. COORDINATE SYSTEM
        // ════════════════════════════════════════════════════════════════════
        h1("7.  Coordinate System"),
        p("All geometry in TEQUILA uses a consistent right-handed coordinate system called the navigation convention:"),
        spacer(60),
        makeTable(
          ["Axis", "Direction", "Represents"],
          [
            ["X", "Right", "Horizontal (left-right in camera view)"],
            ["Y", "Up", "Vertical (gravity opposite)"],
            ["Z", "Toward viewer", "Depth (negative into the scene)"],
          ],
          [1500, 2000, 5860],
          9360
        ),
        spacer(80),
        p("This differs from the standard OpenCV/camera convention where Y points down and Z points into the scene. The conversion is applied by the flip matrix F = diag(1, -1, -1). The world coordinate system is defined by the camera pose on the first frame (T_cum = identity), so the world origin is the camera position when the system starts."),
        spacer(160),

        pageBreak(),

        // ════════════════════════════════════════════════════════════════════
        // 8. VIEWER
        // ════════════════════════════════════════════════════════════════════
        h1("8.  Visualisation"),
        p("TEQUILA uses viser, a browser-based 3D viewer, to display the map and navmesh. The viewer runs as a web server on port 8080 (configurable). No additional software is needed on the viewing device — any modern browser works."),

        h2("8.1  Visual legend"),
        makeTable(
          ["Colour", "Layer name", "What it shows"],
          [
            ["Coloured points", "/scene/map", "Accumulated world map — every point the robot has seen, fused into world space"],
            ["Grey points", "/nav/accum_map", "Downsampled copy of the geometry the latest navmesh was built from"],
            ["Orange points", "/nav/obstacles", "Detected obstacles (height-filtered and denoised)"],
            ["Red points", "/nav/blocked", "Navmesh nodes blocked by obstacles"],
            ["Yellow points", "/nav/free", "Free (passable) navmesh waypoint nodes"],
            ["Blue lines", "/nav/edges", "Passable edges between free nodes"],
            ["Teal line + points", "/nav/path", "A* shortest path to the furthest reachable node"],
            ["Green line + points", "/nav/trajectory", "Full robot trajectory — every camera position since startup"],
          ],
          [2000, 2200, 5160],
          9360
        ),

        h2("8.2  Camera controls"),
        bullet("Left drag — orbit around the scene"),
        bullet("Right drag — pan"),
        bullet("Scroll — zoom in / out"),
        spacer(160),

        pageBreak(),

        // ════════════════════════════════════════════════════════════════════
        // 9. CONFIGURATION
        // ════════════════════════════════════════════════════════════════════
        h1("9.  Configuration Reference"),
        p("All constants are defined in tequila/config.py. CLI flags override the defaults at runtime without changing the file."),

        h2("9.1  Depth model"),
        makeTable(
          ["Constant", "Default", "Description"],
          [
            ["DEPTH_METRIC", "True", "Use Hugging Face metric-depth model (recommended)"],
            ["DEPTH_MODEL_ID", "...Indoor-Small-hf", "HuggingFace model identifier"],
            ["MAX_DEPTH_M", "10.0", "Clip depths beyond this value (metres)"],
            ["FOV_H_DEG", "70.0", "Camera horizontal field of view (degrees)"],
            ["INFER_WIDTH", "1280", "Image width at inference time (pixels)"],
          ],
          [2800, 2200, 4360],
          9360
        ),

        h2("9.2  Flying-pixel removal"),
        makeTable(
          ["Constant", "Default", "Description"],
          [
            ["EDGE_WINDOW_PX", "11", "Erosion kernel size for local-min computation"],
            ["EDGE_THRESHOLD", "0.20", "Max relative depth jump before a pixel is masked"],
          ],
          [2800, 2200, 4360],
          9360
        ),

        h2("9.3  Frame alignment"),
        makeTable(
          ["Constant", "Default", "Description"],
          [
            ["VO_MAX_FEATURES", "2000", "Max ORB keypoints per frame"],
            ["VO_RATIO_TEST", "0.75", "Lowe's ratio test threshold"],
            ["VO_MIN_INLIERS", "12", "Min PnP inliers to accept alignment"],
            ["VO_MIN_SHIFT_M", "0.03", "Min translation to accumulate (below = duplicate view)"],
            ["VO_MAX_SHIFT_M", "2.0", "Max translation per frame (above = sanity failure)"],
            ["VO_MAX_ROT_DEG", "45.0", "Max rotation per frame (degrees)"],
          ],
          [2800, 2200, 4360],
          9360
        ),

        h2("9.4  Navmesh"),
        makeTable(
          ["Constant", "Default", "Description"],
          [
            ["NODE_SPACING", "0.15 m", "Distance between navmesh grid nodes"],
            ["OBS_CLEARANCE_R", "0.40 m", "Obstacle clearance radius — set to at least half robot width"],
            ["OBS_HEIGHT_MIN", "0.18 m", "Min obstacle height above floor (below = floor noise)"],
            ["OBS_HEIGHT_MAX", "0.80 m", "Max obstacle height above floor (above = ceiling / tall furniture)"],
            ["EDGE_MAX_DIST", "0.30 m", "Max edge length between free nodes"],
            ["EDGE_CHECK_STEPS", "20", "Line-of-sight samples per edge"],
          ],
          [2800, 2200, 4360],
          9360
        ),

        h2("9.5  Map accumulation"),
        makeTable(
          ["Constant", "Default", "Description"],
          [
            ["MAP_MAX_DEPTH_M", "4.0 m", "Max depth of points added to display map"],
            ["NAV_ACCUM_MAX_PTS", "500,000", "Hard cap on accumulated cloud size"],
            ["ACCUM_ENABLED", "True", "False = single-frame mode (no accumulation)"],
          ],
          [2800, 2200, 4360],
          9360
        ),
        spacer(160),

        pageBreak(),

        // ════════════════════════════════════════════════════════════════════
        // 10. TROUBLESHOOTING
        // ════════════════════════════════════════════════════════════════════
        h1("10.  Troubleshooting"),
        makeTable(
          ["Symptom", "Root cause", "Fix"],
          [
            ["Very slow (> 30 s per frame)", "No GPU, or Large model on CPU", "Use --width 640 and ensure Small model is set in config.py"],
            ["Floor detected in mid-air", "GPP found a wall or table top; VO drift rotated the cloud", "Handled automatically — RANSAC retries with camera-below height constraint"],
            ["Entire open floor shown as orange obstacles", "Depth noise / VO drift placing floor points within OBS_HEIGHT_MIN of detected plane", "Increase OBS_HEIGHT_MIN in config.py (e.g. 0.25 m)"],
            ["Ceiling or roof shown as obstacle", "VO rotational drift makes ceiling appear close to tilted plane", "Fixed — obstacle height now uses world Y axis, not tilted plane distance"],
            ["Rays or spikes extending behind objects", "Flying pixels at depth discontinuities not fully masked", "Decrease EDGE_THRESHOLD in config.py (e.g. 0.12)"],
            ["Map very sparse or shallow", "Edge mask too aggressive or MAP_MAX_DEPTH too small", "Increase EDGE_THRESHOLD and MAP_MAX_DEPTH_M in config.py"],
            ["Map stretches and fans over time", "Scale drift — likely using relative-depth model", "Set DEPTH_METRIC = True in config.py"],
            ["path=0 nodes in terminal", "No free nodes are connected to the camera start node", "Check OBS_CLEARANCE_R — may be too large for the environment"],
            ["ORB+PnP always failing", "Low-texture environment (plain walls, low light)", "ICP fallback activates automatically; no action needed"],
          ],
          [2400, 3200, 3760],
          9360
        ),
        spacer(160),

        pageBreak(),

        // ════════════════════════════════════════════════════════════════════
        // 11. FILE STRUCTURE
        // ════════════════════════════════════════════════════════════════════
        h1("11.  File Structure"),
        ...codeBlock([
          "  main.py                   Entry point — CLI parsing, model loading, starts viewer",
          "  tequila/",
          "    __init__.py             Package metadata and version",
          "    config.py               All tunable constants",
          "    depth.py                Depth model loading, inference, flying-pixel removal,",
          "                            scale anchoring, back-projection",
          "    pointcloud.py           Voxel downsample, SOR, segmentation helpers",
          "    odometry.py             ORB+PnP visual odometry, ICP fallback",
          "    navmesh.py              GPP floor detection, RANSAC fallback, node grid,",
          "                            obstacle denoising, edge builder, A* planner",
          "    threads.py              CaptureThread, InferenceThread, NavmeshThread,",
          "                            and shared inter-thread queues",
          "    viewer.py               Viser scene update functions and main viewer loop",
          "  depth_anything_v2/        Local DepthAnythingV2 model (relative-depth mode only)",
          "  checkpoints/              Local model weights (relative-depth mode only)",
          "  requirements.txt          Python dependencies",
          "  README.md                 Quick-start guide and reference",
        ]),
        spacer(160),
      ],
    },
  ],
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("C:/Users/shabd/Desktop/Tequila/TEQUILA_Technical_Documentation.docx", buffer);
  console.log("Done.");
});
