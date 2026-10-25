/**
 * export.js
 * ---------------------------------------------------------------------
 * Builds and triggers downloads for the three supported export
 * formats. The JSON schema matches the spec exactly:
 * { shape, rope_length, num_nodes, origin, nodes }
 * ---------------------------------------------------------------------
 */
const Exporter = (() => {

  function buildRecord(shape, ropeLength, numNodes, origin, nodes) {
    return {
      shape,
      rope_length: ropeLength,
      num_nodes: numNodes,
      origin: [origin[0], origin[1]],
      nodes: nodes.map(([x, y]) => [round(x), round(y)])
    };
  }

  function round(v) {
    return Math.round(v * 1e6) / 1e6;
  }

  function toJSON(record) {
    return JSON.stringify(record, null, 4);
  }

  function toCSV(record) {
    const lines = ['node_index,x,y'];
    record.nodes.forEach((p, i) => lines.push(`${i},${p[0]},${p[1]}`));
    return lines.join('\n');
  }

  function toTXT(record) {
    const lines = [
      `shape: ${record.shape}`,
      `rope_length_mm: ${record.rope_length}`,
      `num_nodes: ${record.num_nodes}`,
      `origin: (${record.origin[0]}, ${record.origin[1]})`,
      '',
      'node_index\tx\ty'
    ];
    record.nodes.forEach((p, i) => lines.push(`${i}\t${p[0]}\t${p[1]}`));
    return lines.join('\n');
  }

  function download(filename, content, mime) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function exportJSON(record) {
    download(`rope_${record.shape}.json`, toJSON(record), 'application/json');
  }
  function exportCSV(record) {
    download(`rope_${record.shape}.csv`, toCSV(record), 'text/csv');
  }
  function exportTXT(record) {
    download(`rope_${record.shape}.txt`, toTXT(record), 'text/plain');
  }

  async function exportAllJSON(records) {
    if (typeof JSZip === 'undefined') {
      alert("JSZip library not loaded. Make sure you have internet connection.");
      return;
    }
    const zip = new JSZip();
    for (const r of records) {
      zip.file(`rope_${r.shape}.json`, JSON.stringify(r, null, 4));
    }
    const blob = await zip.generateAsync({ type: 'blob' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'rope_all_shapes.zip';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  return { buildRecord, toJSON, toCSV, toTXT, exportJSON, exportCSV, exportTXT, exportAllJSON };
})();
