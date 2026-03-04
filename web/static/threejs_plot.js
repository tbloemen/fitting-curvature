/**
 * Three.js 2D embedding visualization.
 *
 * Exports: initPlot(containerId), updatePlot(positions, colors, boundary, title)
 */

const ThreeJSPlot = (function () {
  let scene, camera, renderer;
  let pointsGeometry, pointsMesh;
  let boundaryGeometry, boundaryLine;
  let gridMesh = null;
  let axisMesh = null;
  let titleDiv;
  let container;
  let tickLabelDivs = [];
  let lastCurvature = 0;
  let lastProjection = null;
  const frustumSize = 2.4;
  const gridExtent = 1.15;
  const euclideanTicks = [-1, -0.5, 0, 0.5, 1];
  const hyperbolicTicks = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75];
  const ARC_SEGMENTS = 64;

  // Spherical grid settings
  var sphericalMeridians = 12; // number of radial lines (every 30°)
  var boundaryR = 0.95; // matches projection scale in visualisation.py

  // Store last data for SVG export
  let lastPositions = null;
  let lastColors = null;
  let lastBoundaryVisible = false;
  let lastTitle = "";

  /**
   * Compute projected radii for spherical parallels given the projection type.
   * Returns array of {r, label} objects for circles that fit within the view.
   *
   * Uses equally-spaced angles on the sphere (every 30°) and projects them
   * with the correct formula, so the circle spacing reveals the projection's
   * distortion pattern.
   *
   * Stereographic:           R ∝ tan(θ/2)  — circles spread out toward edge
   * Azimuthal equidistant:   R ∝ θ          — equally spaced
   * Orthographic:            R ∝ sin(θ)     — circles compress at edge
   */
  function sphericalGridCircles(projection) {
    // Parallels every 30° from the projection centre.
    // For orthographic (visible hemisphere only) use 15° spacing for more lines.
    var isOrtho = (projection !== "stereographic" && projection !== "azimuthal_equidistant");
    var step = isOrtho ? 15 : 30;
    var maxAngle = isOrtho ? 90 : 150;
    var circles = [];

    for (var deg = step; deg <= maxAngle; deg += step) {
      var theta = deg * Math.PI / 180;
      var r;
      if (projection === "stereographic") {
        r = boundaryR * Math.tan(theta / 2);
      } else if (projection === "azimuthal_equidistant") {
        r = boundaryR * theta / Math.PI;
      } else {
        // orthographic
        r = boundaryR * Math.sin(theta);
      }
      if (r <= 1.05) {
        circles.push({ r: r, label: deg + "\u00B0" });
      }
    }
    return circles;
  }

  /**
   * Compute points along a Poincaré disk geodesic through (a, 0) perpendicular
   * to the x-axis (isVertical=true) or through (0, a) perpendicular to the
   * y-axis (isVertical=false).
   *
   * In the Poincaré disk, geodesics not through the origin are circular arcs
   * orthogonal to the unit boundary circle.  For a vertical geodesic at x = a
   * the arc has centre ((1+a²)/(2a), 0) and radius (1−a²)/(2|a|).
   *
   * Returns array of [x, y] pairs.
   */
  function poincareGeodesicArc(a, isVertical, numSegments) {
    if (Math.abs(a) < 1e-10) {
      // Through origin — a diameter (already a geodesic)
      if (isVertical) return [[0, -1], [0, 1]];
      return [[-1, 0], [1, 0]];
    }

    var a2 = a * a;
    var cx, cy, R;
    // Unit-circle intersection points and reference point on the geodesic
    var p1x, p1y, p2x, p2y, refX, refY;

    if (isVertical) {
      cx = (1 + a2) / (2 * a);  cy = 0;
      p1x = 2 * a / (1 + a2);   p1y =  (1 - a2) / (1 + a2);
      p2x = p1x;                 p2y = -p1y;
      refX = a;                  refY = 0;
    } else {
      cx = 0;                    cy = (1 + a2) / (2 * a);
      p1x =  (1 - a2) / (1 + a2); p1y = 2 * a / (1 + a2);
      p2x = -p1x;                  p2y = p1y;
      refX = 0;                     refY = a;
    }
    R = Math.abs(1 - a2) / (2 * Math.abs(a));

    // Angles (relative to circle centre) of the two boundary points and the
    // reference point that the arc must pass through.
    var alpha1   = Math.atan2(p1y - cy, p1x - cx);
    var alpha2   = Math.atan2(p2y - cy, p2x - cx);
    var alphaRef = Math.atan2(refY - cy, refX - cx);

    // Normalise alpha1 and alpha2 into the window [alphaRef − π, alphaRef + π)
    // so the sweep from min to max is guaranteed to contain alphaRef.
    function normAngle(angle, centre) {
      while (angle < centre - Math.PI) angle += 2 * Math.PI;
      while (angle >= centre + Math.PI) angle -= 2 * Math.PI;
      return angle;
    }
    alpha1 = normAngle(alpha1, alphaRef);
    alpha2 = normAngle(alpha2, alphaRef);

    var startAngle = Math.min(alpha1, alpha2);
    var endAngle   = Math.max(alpha1, alpha2);

    var points = [];
    for (var i = 0; i <= numSegments; i++) {
      var t = startAngle + (endAngle - startAngle) * i / numSegments;
      points.push([cx + R * Math.cos(t), cy + R * Math.sin(t)]);
    }
    return points;
  }

  /**
   * Build grid + axis geometry for the current curvature and projection.
   */
  function buildGrid(k, projection) {
    // Remove old meshes
    if (gridMesh) { scene.remove(gridMesh); gridMesh = null; }
    if (axisMesh) { scene.remove(axisMesh); axisMesh = null; }

    var isHyperbolic = k < 0;
    var isSpherical = k > 0;

    // --- Grid lines ---
    var gridVerts = [];

    if (isSpherical) {
      // Concentric circles (parallels) with projection-correct spacing
      var circles = sphericalGridCircles(projection);
      for (var ci = 0; ci < circles.length; ci++) {
        var r = circles[ci].r;
        for (var i = 0; i < ARC_SEGMENTS; i++) {
          var t1 = (2 * Math.PI * i) / ARC_SEGMENTS;
          var t2 = (2 * Math.PI * (i + 1)) / ARC_SEGMENTS;
          gridVerts.push(
            r * Math.cos(t1), r * Math.sin(t1), -0.2,
            r * Math.cos(t2), r * Math.sin(t2), -0.2
          );
        }
      }

      // Radial meridians (out to outermost visible circle)
      var maxR = circles.length > 0 ? circles[circles.length - 1].r : boundaryR;
      for (var mi = 0; mi < sphericalMeridians; mi++) {
        var angle = (2 * Math.PI * mi) / sphericalMeridians;
        var dx = Math.cos(angle);
        var dy = Math.sin(angle);
        gridVerts.push(0, 0, -0.2, dx * maxR, dy * maxR, -0.2);
      }
    } else if (isHyperbolic) {
      var ticks = hyperbolicTicks;
      for (var ti = 0; ti < ticks.length; ti++) {
        var v = ticks[ti];
        if (Math.abs(v) < 1e-10) continue;
        // Horizontal geodesic at y = v
        var hPts = poincareGeodesicArc(v, false, ARC_SEGMENTS);
        for (var i = 0; i < hPts.length - 1; i++) {
          gridVerts.push(hPts[i][0], hPts[i][1], -0.2);
          gridVerts.push(hPts[i + 1][0], hPts[i + 1][1], -0.2);
        }
        // Vertical geodesic at x = v
        var vPts = poincareGeodesicArc(v, true, ARC_SEGMENTS);
        for (var i = 0; i < vPts.length - 1; i++) {
          gridVerts.push(vPts[i][0], vPts[i][1], -0.2);
          gridVerts.push(vPts[i + 1][0], vPts[i + 1][1], -0.2);
        }
      }
    } else {
      // Euclidean: straight lines
      var ticks = euclideanTicks;
      for (var ti = 0; ti < ticks.length; ti++) {
        var v = ticks[ti];
        if (Math.abs(v) < 1e-10) continue;
        gridVerts.push(-gridExtent, v, -0.2, gridExtent, v, -0.2);
        gridVerts.push(v, -gridExtent, -0.2, v, gridExtent, -0.2);
      }
    }

    if (gridVerts.length > 0) {
      var gridGeom = new THREE.BufferGeometry();
      gridGeom.setAttribute("position", new THREE.Float32BufferAttribute(gridVerts, 3));
      gridMesh = new THREE.LineSegments(gridGeom, new THREE.LineBasicMaterial({ color: 0xdddddd }));
      scene.add(gridMesh);
    }

    // --- Axis lines ---
    var axisVerts = [];
    if (isSpherical) {
      // Two perpendicular diameters as axes, extending to outermost circle
      var circles = sphericalGridCircles(projection);
      var axisR = circles.length > 0 ? circles[circles.length - 1].r : boundaryR;
      axisVerts.push(
        -axisR, 0, -0.1, axisR, 0, -0.1,
         0, -axisR, -0.1, 0, axisR, -0.1
      );
    } else {
      axisVerts.push(
        -gridExtent, 0, -0.1, gridExtent, 0, -0.1,
        0, -gridExtent, -0.1, 0, gridExtent, -0.1
      );
    }
    var axisGeom = new THREE.BufferGeometry();
    axisGeom.setAttribute("position", new THREE.Float32BufferAttribute(axisVerts, 3));
    axisMesh = new THREE.LineSegments(axisGeom, new THREE.LineBasicMaterial({ color: 0x999999 }));
    scene.add(axisMesh);

    // --- Tick labels ---
    for (var i = 0; i < tickLabelDivs.length; i++) {
      tickLabelDivs[i].parentNode.removeChild(tickLabelDivs[i]);
    }
    tickLabelDivs = [];

    if (isSpherical) {
      // Place each label at a different angle along its circle so they
      // fan out and never overlap, even when circles are close together.
      var circles = sphericalGridCircles(projection);
      // Labels sit between two meridians; spread across 45° arc in top-right
      var labelAngleStart = 10 * Math.PI / 180;
      var labelAngleStep = circles.length > 1
        ? (35 * Math.PI / 180) / (circles.length - 1)
        : 0;
      for (var ci = 0; ci < circles.length; ci++) {
        var phi = labelAngleStart + ci * labelAngleStep;
        var label = document.createElement("div");
        label.style.cssText = "position:absolute;font-size:10px;color:#999;pointer-events:none;font-family:Arial,sans-serif;";
        label.textContent = circles[ci].label;
        label._worldX = circles[ci].r * Math.cos(phi);
        label._worldY = circles[ci].r * Math.sin(phi);
        label._offsetX = 3;
        label._offsetY = -12;
        container.appendChild(label);
        tickLabelDivs.push(label);
      }
    } else {
      var labelTicks = isHyperbolic ? [-0.75, -0.5, -0.25, 0.25, 0.5, 0.75] : [-1, -0.5, 0.5, 1];
      for (var ti = 0; ti < labelTicks.length; ti++) {
        var v = labelTicks[ti];
        var xLabel = document.createElement("div");
        xLabel.style.cssText = "position:absolute;font-size:10px;color:#999;pointer-events:none;font-family:Arial,sans-serif;";
        xLabel.textContent = v.toString();
        xLabel._worldX = v;
        xLabel._worldY = 0;
        xLabel._offsetX = -5;
        xLabel._offsetY = 10;
        container.appendChild(xLabel);
        tickLabelDivs.push(xLabel);

        var yLabel = document.createElement("div");
        yLabel.style.cssText = "position:absolute;font-size:10px;color:#999;pointer-events:none;font-family:Arial,sans-serif;";
        yLabel.textContent = v.toString();
        yLabel._worldX = 0;
        yLabel._worldY = v;
        yLabel._offsetX = 5;
        yLabel._offsetY = -5;
        container.appendChild(yLabel);
        tickLabelDivs.push(yLabel);
      }
    }
    updateTickPositions();
  }

  function initPlot(containerId) {
    container = document.getElementById(containerId);
    if (!container) {
      console.error("Container not found:", containerId);
      return;
    }

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);

    // Orthographic camera for 2D
    const aspect = container.clientWidth / container.clientHeight;
    camera = new THREE.OrthographicCamera(
      (frustumSize * aspect) / -2,
      (frustumSize * aspect) / 2,
      frustumSize / 2,
      frustumSize / -2,
      0.1,
      1000,
    );
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.domElement.style.display = "block";
    container.appendChild(renderer.domElement);

    // Circle texture for round points
    const circleCanvas = document.createElement("canvas");
    circleCanvas.width = 64;
    circleCanvas.height = 64;
    const ctx = circleCanvas.getContext("2d");
    ctx.beginPath();
    ctx.arc(32, 32, 30, 0, 2 * Math.PI);
    ctx.fillStyle = "white";
    ctx.fill();
    const circleTexture = new THREE.CanvasTexture(circleCanvas);

    // Points
    pointsGeometry = new THREE.BufferGeometry();
    const pointsMaterial = new THREE.PointsMaterial({
      size: 8,
      vertexColors: true,
      sizeAttenuation: false,
      map: circleTexture,
      alphaTest: 0.5,
      transparent: true,
    });
    pointsMesh = new THREE.Points(pointsGeometry, pointsMaterial);
    scene.add(pointsMesh);

    // Build initial grid (Euclidean)
    buildGrid(0, null);

    // Boundary line (dashed)
    boundaryGeometry = new THREE.BufferGeometry();
    const boundaryMaterial = new THREE.LineDashedMaterial({
      color: 0x000000,
      linewidth: 1,
      dashSize: 0.05,
      gapSize: 0.03,
    });
    boundaryLine = new THREE.Line(boundaryGeometry, boundaryMaterial);
    boundaryLine.visible = false;
    scene.add(boundaryLine);

    // Title overlay
    titleDiv = document.createElement("div");
    titleDiv.style.cssText =
      "position:absolute;top:10px;left:10px;color:#333;font-family:Arial,sans-serif;" +
      "font-size:14px;font-weight:bold;pointer-events:none;white-space:pre-line;z-index:10;";
    container.appendChild(titleDiv);
    titleDiv.textContent = "Waiting for training to start...";

    // Animation loop
    function animate() {
      requestAnimationFrame(animate);
      renderer.render(scene, camera);
    }
    animate();

    // Resize handler
    window.addEventListener("resize", function () {
      if (!container.clientWidth || !container.clientHeight) return;
      const aspect = container.clientWidth / container.clientHeight;
      camera.left = (frustumSize * aspect) / -2;
      camera.right = (frustumSize * aspect) / 2;
      camera.top = frustumSize / 2;
      camera.bottom = frustumSize / -2;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
      updateTickPositions();
    });
  }

  function updateTickPositions() {
    if (!camera || !renderer) return;
    const w = renderer.domElement.clientWidth;
    const h = renderer.domElement.clientHeight;
    for (const label of tickLabelDivs) {
      const v = new THREE.Vector3(label._worldX, label._worldY, 0);
      v.project(camera);
      const px = (v.x * 0.5 + 0.5) * w + label._offsetX;
      const py = (-v.y * 0.5 + 0.5) * h + label._offsetY;
      label.style.left = px + "px";
      label.style.top = py + "px";
    }
  }

  function setCurvature(k, projection) {
    projection = projection || null;
    if (k === lastCurvature && projection === lastProjection) return;
    lastCurvature = k;
    lastProjection = projection;
    if (scene) buildGrid(k, projection);
  }

  /**
   * Update embedding points.
   */
  function updatePlot(positions, colors, boundary, title) {
    if (!pointsGeometry) return;

    // Store for SVG export
    if (positions && positions.length > 0) {
      lastPositions = positions;
      lastColors = colors;
    }
    if (title) lastTitle = title;

    if (positions && positions.length > 0) {
      // Convert 2D positions to 3D (add z=0)
      const n = positions.length / 2;
      const pos3d = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        pos3d[i * 3] = positions[i * 2];
        pos3d[i * 3 + 1] = positions[i * 2 + 1];
        pos3d[i * 3 + 2] = 0;
      }

      pointsGeometry.setAttribute(
        "position",
        new THREE.BufferAttribute(pos3d, 3),
      );
      pointsGeometry.setAttribute(
        "color",
        new THREE.BufferAttribute(colors, 3),
      );
      pointsGeometry.attributes.position.needsUpdate = true;
      pointsGeometry.attributes.color.needsUpdate = true;
      pointsGeometry.computeBoundingSphere();
    }

    // Boundary
    if (boundary && boundary.length > 0) {
      const n = boundary.length / 2;
      const bnd3d = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        bnd3d[i * 3] = boundary[i * 2];
        bnd3d[i * 3 + 1] = boundary[i * 2 + 1];
        bnd3d[i * 3 + 2] = 0;
      }
      boundaryGeometry.setAttribute(
        "position",
        new THREE.BufferAttribute(bnd3d, 3),
      );
      boundaryGeometry.attributes.position.needsUpdate = true;
      boundaryLine.computeLineDistances();
      boundaryLine.visible = true;
    }

    if (title && titleDiv) {
      titleDiv.textContent = title;
    }
  }

  function setBoundary(boundaryPoints) {
    if (!boundaryGeometry) return;
    if (boundaryPoints && boundaryPoints.length > 0) {
      const n = boundaryPoints.length / 2;
      const bnd3d = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        bnd3d[i * 3] = boundaryPoints[i * 2];
        bnd3d[i * 3 + 1] = boundaryPoints[i * 2 + 1];
        bnd3d[i * 3 + 2] = 0;
      }
      boundaryGeometry.setAttribute(
        "position",
        new THREE.BufferAttribute(bnd3d, 3),
      );
      boundaryGeometry.attributes.position.needsUpdate = true;
      boundaryLine.computeLineDistances();
      boundaryLine.visible = true;
      lastBoundaryVisible = true;
    } else {
      boundaryLine.visible = false;
      lastBoundaryVisible = false;
    }
  }

  function hideBoundary() {
    if (boundaryLine) boundaryLine.visible = false;
    lastBoundaryVisible = false;
  }

  /**
   * Generate SVG elements for grid lines, adapting to geometry type.
   */
  function svgGridElements(k, pad, projection) {
    var parts = [];
    var isHyperbolic = k < 0;
    var isSpherical = k > 0;

    if (isSpherical) {
      var circles = sphericalGridCircles(projection);
      // Concentric circles
      for (var ci = 0; ci < circles.length; ci++) {
        parts.push(
          '<circle cx="0" cy="0" r="' + circles[ci].r.toFixed(5) + '" fill="none" stroke="#ddd" stroke-width="0.005"/>'
        );
      }
      // Radial meridians
      var maxR = circles.length > 0 ? circles[circles.length - 1].r : boundaryR;
      for (var mi = 0; mi < sphericalMeridians; mi++) {
        var angle = (2 * Math.PI * mi) / sphericalMeridians;
        var ex = (Math.cos(angle) * maxR).toFixed(5);
        var ey = (-Math.sin(angle) * maxR).toFixed(5); // SVG y-flip
        parts.push(
          '<line x1="0" y1="0" x2="' + ex + '" y2="' + ey + '" stroke="#ddd" stroke-width="0.005"/>'
        );
      }
    } else if (isHyperbolic) {
      var ticks = hyperbolicTicks;
      for (var ti = 0; ti < ticks.length; ti++) {
        var v = ticks[ti];
        if (Math.abs(v) < 1e-10) continue;
        // Horizontal geodesic (SVG y is flipped → negate y)
        var hPts = poincareGeodesicArc(v, false, ARC_SEGMENTS);
        var hStr = "";
        for (var i = 0; i < hPts.length; i++) {
          hStr += hPts[i][0].toFixed(5) + "," + (-hPts[i][1]).toFixed(5);
          if (i < hPts.length - 1) hStr += " ";
        }
        parts.push('<polyline points="' + hStr + '" fill="none" stroke="#ddd" stroke-width="0.005"/>');

        // Vertical geodesic
        var vPts = poincareGeodesicArc(v, true, ARC_SEGMENTS);
        var vStr = "";
        for (var i = 0; i < vPts.length; i++) {
          vStr += vPts[i][0].toFixed(5) + "," + (-vPts[i][1]).toFixed(5);
          if (i < vPts.length - 1) vStr += " ";
        }
        parts.push('<polyline points="' + vStr + '" fill="none" stroke="#ddd" stroke-width="0.005"/>');
      }
    } else {
      var ticks = euclideanTicks;
      for (var ti = 0; ti < ticks.length; ti++) {
        var v = ticks[ti];
        if (Math.abs(v) < 1e-10) continue;
        parts.push(
          '<line x1="' + (-pad) + '" y1="' + v + '" x2="' + pad + '" y2="' + v + '" stroke="#ddd" stroke-width="0.005"/>'
        );
        parts.push(
          '<line x1="' + v + '" y1="' + (-pad) + '" x2="' + v + '" y2="' + pad + '" stroke="#ddd" stroke-width="0.005"/>'
        );
      }
    }
    return parts;
  }

  /**
   * Generate SVG axis lines.
   */
  function svgAxisElements(k, pad, projection) {
    var parts = [];
    if (k > 0) {
      var circles = sphericalGridCircles(projection);
      var axisR = circles.length > 0 ? circles[circles.length - 1].r : boundaryR;
      parts.push(
        '<line x1="' + (-axisR) + '" y1="0" x2="' + axisR + '" y2="0" stroke="#999" stroke-width="0.008"/>'
      );
      parts.push(
        '<line x1="0" y1="' + (-axisR) + '" x2="0" y2="' + axisR + '" stroke="#999" stroke-width="0.008"/>'
      );
    } else {
      parts.push(
        '<line x1="' + (-pad) + '" y1="0" x2="' + pad + '" y2="0" stroke="#999" stroke-width="0.008"/>'
      );
      parts.push(
        '<line x1="0" y1="' + (-pad) + '" x2="0" y2="' + pad + '" stroke="#999" stroke-width="0.008"/>'
      );
    }
    return parts;
  }

  /**
   * Generate SVG tick label elements.
   */
  function svgTickElements(k, projection) {
    var parts = [];
    if (k > 0) {
      var circles = sphericalGridCircles(projection);
      var labelAngleStart = 10 * Math.PI / 180;
      var labelAngleStep = circles.length > 1
        ? (35 * Math.PI / 180) / (circles.length - 1)
        : 0;
      for (var ci = 0; ci < circles.length; ci++) {
        var phi = labelAngleStart + ci * labelAngleStep;
        var lx = (circles[ci].r * Math.cos(phi) + 0.02).toFixed(5);
        var ly = (-circles[ci].r * Math.sin(phi) - 0.01).toFixed(5); // SVG y-flip
        parts.push(
          '<text x="' + lx + '" y="' + ly + '" text-anchor="start" font-size="0.06" fill="#999" font-family="Arial,sans-serif">' + circles[ci].label + '</text>'
        );
      }
    } else {
      var isHyperbolic = k < 0;
      var labelTicks = isHyperbolic ? [-0.75, -0.5, -0.25, 0.25, 0.5, 0.75] : [-1, -0.5, 0.5, 1];
      for (var ti = 0; ti < labelTicks.length; ti++) {
        var v = labelTicks[ti];
        parts.push(
          '<text x="' + v + '" y="0.07" text-anchor="middle" font-size="0.06" fill="#999" font-family="Arial,sans-serif">' + v + '</text>'
        );
        parts.push(
          '<text x="0.04" y="' + (-v + 0.02) + '" text-anchor="start" font-size="0.06" fill="#999" font-family="Arial,sans-serif">' + v + '</text>'
        );
      }
    }
    return parts;
  }

  function downloadSVG(dataset, curvature) {
    if (!lastPositions || lastPositions.length === 0) return;

    var k = (curvature !== undefined && curvature !== null) ? curvature : 0;
    const n = lastPositions.length / 2;
    const pad = 1.15;
    const parts = [];

    parts.push('<?xml version="1.0" encoding="UTF-8"?>');
    parts.push(
      '<svg xmlns="http://www.w3.org/2000/svg"' +
        ' viewBox="' + (-pad) + " " + (-pad) + " " + (pad * 2) + " " + (pad * 2) + '"' +
        ' width="500" height="500">',
    );

    // White background
    parts.push(
      '<rect x="' + (-pad) + '" y="' + (-pad) + '" width="' + (pad * 2) + '" height="' + (pad * 2) + '" fill="white"/>',
    );

    // Grid lines
    var gridParts = svgGridElements(k, pad, lastProjection);
    for (var i = 0; i < gridParts.length; i++) parts.push(gridParts[i]);

    // Axis lines
    var axisParts = svgAxisElements(k, pad, lastProjection);
    for (var i = 0; i < axisParts.length; i++) parts.push(axisParts[i]);

    // Tick labels
    var tickParts = svgTickElements(k, lastProjection);
    for (var i = 0; i < tickParts.length; i++) parts.push(tickParts[i]);

    // Boundary ring
    if (lastBoundaryVisible) {
      parts.push(
        '<circle cx="0" cy="0" r="1" fill="none" stroke="#444"' +
          ' stroke-width="0.015" stroke-dasharray="0.05 0.03"/>',
      );
    }

    // Points (SVG y-axis is flipped relative to data coords)
    for (let i = 0; i < n; i++) {
      const x = lastPositions[i * 2].toFixed(5);
      const y = (-lastPositions[i * 2 + 1]).toFixed(5); // flip y
      const r = Math.round(lastColors[i * 3] * 255);
      const g = Math.round(lastColors[i * 3 + 1] * 255);
      const b = Math.round(lastColors[i * 3 + 2] * 255);
      parts.push(
        '<circle cx="' + x + '" cy="' + y + '" r="0.016"' +
          ' fill="rgb(' + r + "," + g + "," + b + ')" opacity="0.85"/>',
      );
    }

    parts.push("</svg>");

    var ds = (dataset || "embedding").replace(/[^a-zA-Z0-9_-]/g, "_");
    var filename = ds + "_k" + k + ".svg";

    const blob = new Blob([parts.join("\n")], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  return { initPlot, updatePlot, setBoundary, hideBoundary, setCurvature, downloadSVG };
})();
