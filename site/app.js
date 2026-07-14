const search = document.querySelector('[data-search]');
const cards = [...document.querySelectorAll('[data-card]')];

if (search) {
  search.addEventListener('input', () => {
    const query = search.value.trim().toLocaleLowerCase();
    for (const card of cards) {
      card.hidden = query && !card.textContent.toLocaleLowerCase().includes(query);
    }
  });
}
