/**
 * Three.js embedding visualization
 *
 * This script initializes a Three.js scene for rendering 2D embeddings
 * with WebGL for high-performance visualization.
 *
 * Template variables:
 * - {{PLOT_ID}}: Unique identifier for this plot instance
 */

(function() {
    const containerId = 'container_{{PLOT_ID}}';
    const container = document.getElementById(containerId);

    if (!container) {
        console.error('Container not found:', containerId);
        return;
    }

    console.log('ThreeJS init - Container found:', container);
    console.log('ThreeJS init - Container size:', container.clientWidth, 'x', container.clientHeight);

    // Wait for Three.js to load
    function initThreeJS() {
        if (typeof THREE === 'undefined') {
            console.log('Waiting for THREE.js to load...');
            setTimeout(initThreeJS, 100);
            return;
        }

        console.log('THREE.js loaded, initializing scene');

        // Ensure container has size
        if (container.clientWidth === 0 || container.clientHeight === 0) {
            console.warn('Container has zero size, setting fallback dimensions');
            container.style.width = '700px';
            container.style.height = '600px';
        }

        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xffffff);

        // Camera setup (orthographic for 2D view)
        const aspect = container.clientWidth / container.clientHeight;
        const frustumSize = 2.4;  // View range from -1.2 to 1.2
        const camera = new THREE.OrthographicCamera(
            frustumSize * aspect / -2,
            frustumSize * aspect / 2,
            frustumSize / 2,
            frustumSize / -2,
            0.1,
            1000
        );
        camera.position.set(0, 0, 5);
        camera.lookAt(0, 0, 0);

        // Renderer setup
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);

        // Style the canvas to ensure it's visible
        renderer.domElement.style.display = 'block';
        renderer.domElement.style.position = 'absolute';
        renderer.domElement.style.top = '0';
        renderer.domElement.style.left = '0';

        container.appendChild(renderer.domElement);

        console.log('Renderer created, canvas size:', renderer.domElement.width, 'x', renderer.domElement.height);
        console.log('Canvas appended to container');
        console.log('Canvas element:', renderer.domElement);

        // Points geometry
        const pointsGeometry = new THREE.BufferGeometry();
        const pointsMaterial = new THREE.PointsMaterial({
            size: 8,
            vertexColors: true,
            sizeAttenuation: false
        });
        const pointsMesh = new THREE.Points(pointsGeometry, pointsMaterial);
        scene.add(pointsMesh);

        // Boundary circle (initially hidden)
        const boundaryGeometry = new THREE.BufferGeometry();
        const boundaryMaterial = new THREE.LineDashedMaterial({
            color: 0x000000,
            linewidth: 1,
            dashSize: 0.05,
            gapSize: 0.03
        });
        const boundaryLine = new THREE.Line(boundaryGeometry, boundaryMaterial);
        boundaryLine.visible = false;
        scene.add(boundaryLine);

        // Title text (rendered via HTML overlay)
        const titleDiv = document.createElement('div');
        titleDiv.id = 'title_{{PLOT_ID}}';
        titleDiv.style.position = 'absolute';
        titleDiv.style.top = '10px';
        titleDiv.style.left = '10px';
        titleDiv.style.color = '#333';
        titleDiv.style.fontFamily = 'Arial, sans-serif';
        titleDiv.style.fontSize = '14px';
        titleDiv.style.fontWeight = 'bold';
        titleDiv.style.pointerEvents = 'none';
        titleDiv.style.whiteSpace = 'pre-line';
        titleDiv.style.zIndex = '10';  // Ensure title is above canvas
        container.style.position = 'relative';
        container.appendChild(titleDiv);

        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }
        animate();
        console.log('Animation loop started');

        // Window resize handler
        function onWindowResize() {
            const aspect = container.clientWidth / container.clientHeight;
            camera.left = frustumSize * aspect / -2;
            camera.right = frustumSize * aspect / 2;
            camera.top = frustumSize / 2;
            camera.bottom = frustumSize / -2;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }
        window.addEventListener('resize', onWindowResize);

        // Update function (called from Python)
        window.updateEmbedding_{{PLOT_ID}} = function(positions, colors, boundary, title) {
            console.log('updateEmbedding called - positions:', positions ? positions.length : 0,
                        'colors:', colors ? colors.length : 0,
                        'boundary:', boundary ? boundary.length : 0);

            // Update points
            if (positions && positions.length > 0) {
                const positionsArray = new Float32Array(positions);
                const colorsArray = new Float32Array(colors);

                pointsGeometry.setAttribute('position',
                    new THREE.BufferAttribute(positionsArray, 3));
                pointsGeometry.setAttribute('color',
                    new THREE.BufferAttribute(colorsArray, 3));
                pointsGeometry.attributes.position.needsUpdate = true;
                pointsGeometry.attributes.color.needsUpdate = true;
                pointsGeometry.computeBoundingSphere();

                console.log('Points updated:', positions.length / 3, 'points');
            } else {
                console.log('No positions to update');
            }

            // Update boundary
            if (boundary && boundary.length > 0) {
                const boundaryArray = new Float32Array(boundary);
                boundaryGeometry.setAttribute('position',
                    new THREE.BufferAttribute(boundaryArray, 3));
                boundaryGeometry.attributes.position.needsUpdate = true;
                boundaryLine.computeLineDistances();
                boundaryLine.visible = true;
            } else {
                boundaryLine.visible = false;
            }

            // Update title
            if (title) {
                titleDiv.textContent = title;
            }
        };

        // Show empty state message
        console.log('Setting initial empty state message');
        window.updateEmbedding_{{PLOT_ID}}([], [], null, 'Waiting for training to start...');
    }

    initThreeJS();
    console.log('Initialization function called');
})();
