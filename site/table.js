
(function () {
  function cellValue(row, index) {
    var cell = row.children[index];
    return cell ? (cell.getAttribute("data-sort") || cell.textContent.trim()) : "";
  }
  function comparator(index, numeric, dir) {
    return function (a, b) {
      var x = cellValue(a, index), y = cellValue(b, index);
      var r;
      if (numeric) {
        var nx = parseFloat(x), ny = parseFloat(y);
        var xb = isNaN(nx), yb = isNaN(ny);
        if (xb && yb) r = 0; else if (xb) r = 1; else if (yb) r = -1; else r = nx - ny;
      } else {
        r = x.localeCompare(y, undefined, { numeric: true, sensitivity: "base" });
      }
      return dir === "descending" ? -r : r;
    };
  }
  document.querySelectorAll("table[data-sortable]").forEach(function (table) {
    var head = table.tHead, body = table.tBodies[0];
    if (!head || !body) return;
    var headers = Array.prototype.slice.call(head.rows[0].cells);
    headers.forEach(function (th, index) {
      var btn = th.querySelector(".sort-btn");
      if (!btn) return;
      btn.addEventListener("click", function () {
        var numeric = th.getAttribute("data-type") === "number";
        var dir = th.getAttribute("aria-sort") === "ascending" ? "descending" : "ascending";
        headers.forEach(function (other) { other.removeAttribute("aria-sort"); });
        th.setAttribute("aria-sort", dir);
        var rows = Array.prototype.slice.call(body.rows);
        rows.sort(comparator(index, numeric, dir));
        rows.forEach(function (row) { body.appendChild(row); });
      });
    });
  });
})();
