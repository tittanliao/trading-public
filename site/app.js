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
    for (const row of document.querySelectorAll('[data-view-table] tbody tr')) {
      row.hidden = query && !row.textContent.toLocaleLowerCase().includes(query);
    }
  });
}

/* ---------------------------------------------------------------- view switch */

const VIEW_KEY = 'trading-public:view';
const toggle = document.querySelector('[data-view-toggle]');
const cardsView = document.querySelector('[data-view-cards]');
const tableView = document.querySelector('[data-view-table]');

function applyView(view) {
  if (!cardsView || !tableView) return;
  const table = view === 'table';
  cardsView.hidden = table;
  tableView.hidden = !table;
  for (const button of toggle ? toggle.querySelectorAll('button') : []) {
    button.setAttribute('aria-pressed', String(button.dataset.view === view));
  }
}

if (toggle && cardsView && tableView) {
  // Reading storage can throw outright in a private window or with site data blocked, so
  // the preference is best-effort and the page must render correctly without it.
  let saved = null;
  try { saved = localStorage.getItem(VIEW_KEY); } catch (error) { saved = null; }
  applyView(saved === 'table' ? 'table' : 'cards');

  toggle.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-view]');
    if (!button) return;
    applyView(button.dataset.view);
    try { localStorage.setItem(VIEW_KEY, button.dataset.view); } catch (error) { /* ignore */ }
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
