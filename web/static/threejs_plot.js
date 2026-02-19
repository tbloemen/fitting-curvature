/**
 * Three.js 2D embedding visualization.
 *
 * Exports: initPlot(containerId), updatePlot(positions, colors, boundary, title)
 */

const ThreeJSPlot = (function () {
    let scene, camera, renderer;
    let pointsGeometry, pointsMesh;
    let boundaryGeometry, boundaryLine;
    let titleDiv;
    let container;
    const frustumSize = 2.4;

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
            1000
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
        });
    }

    /**
     * Update embedding points.
     * @param {Float32Array} positions - flat [x0,y0,x1,y1,...] (2 floats per point)
     * @param {Float32Array} colors - flat [r0,g0,b0,r1,g1,b1,...] (3 floats per point)
     * @param {Float32Array|null} boundary - flat [x0,y0,x1,y1,...] boundary circle points
     * @param {string} title - title text
     */
    function updatePlot(positions, colors, boundary, title) {
        if (!pointsGeometry) return;

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
                new THREE.BufferAttribute(pos3d, 3)
            );
            pointsGeometry.setAttribute(
                "color",
                new THREE.BufferAttribute(colors, 3)
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
                new THREE.BufferAttribute(bnd3d, 3)
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
                new THREE.BufferAttribute(bnd3d, 3)
            );
            boundaryGeometry.attributes.position.needsUpdate = true;
            boundaryLine.computeLineDistances();
            boundaryLine.visible = true;
        } else {
            boundaryLine.visible = false;
        }
    }

    function hideBoundary() {
        if (boundaryLine) boundaryLine.visible = false;
    }

    return { initPlot, updatePlot, setBoundary, hideBoundary };
})();
