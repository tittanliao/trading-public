// Filter, view switch, and table sorting. Progressive: every page works without this file,
// and the table exists in the HTML rather than being built here, so a reader with scripting
// disabled still gets a usable page — they simply cannot switch between the two layouts.

const search = document.querySelector('[data-search]');
const cards = [...document.querySelectorAll('[data-card]')];

if (search) {
  search.addEventListener('input', () => {
    const query = search.value.trim().toLocaleLowerCase();
    for (const card of cards) {
      card.hidden = query && !card.textContent.toLocaleLowerCase().includes(query);
    }
    // The table carries the same studies, so a filter that only moved the cards would
    // silently stop working the moment someone switched view.
    for (const row of document.querySelectorAll('[data-study-table] tbody tr')) {
      row.hidden = query && !row.textContent.toLocaleLowerCase().includes(query);
    }
  });
}

/* ---------------------------------------------------------------- table sorting */

for (const table of document.querySelectorAll('table[data-sortable]')) {
  const body = table.tBodies[0];
  if (!body) continue;
  const headers = [...table.querySelectorAll('th[data-sort]')];

  headers.forEach((header, column) => {
    header.addEventListener('click', () => {
      const ascending = header.getAttribute('aria-sort') !== 'ascending';
      for (const other of headers) other.removeAttribute('aria-sort');
      header.setAttribute('aria-sort', ascending ? 'ascending' : 'descending');

      const rows = [...body.rows];
      rows.sort((left, right) => {
        const a = left.cells[column]?.textContent.trim() ?? '';
        const b = right.cells[column]?.textContent.trim() ?? '';
        // Compare as numbers when both sides are numeric, so "9" sorts below "10" rather
        // than above it the way a string comparison would put it.
        const na = Number.parseFloat(a.replace(/[^\d.-]/g, ''));
        const nb = Number.parseFloat(b.replace(/[^\d.-]/g, ''));
        const numeric = !Number.isNaN(na) && !Number.isNaN(nb) && /\d/.test(a) && /\d/.test(b);
        const order = numeric ? na - nb : a.localeCompare(b, 'en');
        return ascending ? order : -order;
      });
      for (const row of rows) body.append(row);
    });
  });
}
