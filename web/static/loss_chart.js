/**
 * Plotly loss chart.
 *
 * Exports: LossChart.init(containerId), LossChart.addPoint(iteration, loss)
 */

const LossChart = (function () {
    let containerId;
    const iterations = [];
    const losses = [];

    function init(id) {
        containerId = id;
        const layout = {
            title: "Loss History",
            xaxis: { title: "Iteration" },
            yaxis: { title: "Loss", type: "log" },
            template: "plotly_white",
            hovermode: "closest",
            margin: { t: 40, r: 20, b: 50, l: 60 },
        };
        const trace = {
            x: [],
            y: [],
            mode: "lines+markers",
            name: "Loss",
            line: { color: "#1f77b4", width: 2 },
            marker: { size: 4 },
            hovertemplate: "Iteration: %{x}<br>Loss: %{y:.4f}<extra></extra>",
        };
        Plotly.newPlot(containerId, [trace], layout, { responsive: true });
    }

    function addPoint(iteration, loss) {
        iterations.push(iteration);
        losses.push(loss);
        Plotly.extendTraces(containerId, { x: [[iteration]], y: [[loss]] }, [0]);
    }

    function reset() {
        iterations.length = 0;
        losses.length = 0;
        const container = document.getElementById(containerId);
        if (container) {
            Plotly.purge(containerId);
            init(containerId);
        }
    }

    return { init, addPoint, reset };
})();
