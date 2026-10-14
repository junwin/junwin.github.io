/* Minimal dependency-free photo lightbox for the /gallery/ page.
   Data comes from gallery.json (same format the image_description
   gallery-builder writes: entries with "file" and "alt"). */
(function () {
  var grid = document.getElementById('photo-grid');
  if (!grid) return;

  var images = [];
  var current = 0;
  var lightbox, lightboxImg, lightboxCaption;

  fetch('gallery.json')
    .then(function (r) { if (!r.ok) throw new Error('gallery.json not found'); return r.json(); })
    .then(build)
    .catch(function () {
      var p = document.createElement('p');
      p.className = 'small';
      p.textContent = 'No photos available right now.';
      grid.replaceWith(p);
    });

  function build(entries) {
    images = entries.map(function (e) {
      return { src: e.file, alt: e.alt || '' };
    });
    grid.innerHTML = '';
    images.forEach(function (item, i) {
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'thumb';
      button.setAttribute('aria-label', 'Open photo ' + (i + 1) + ' of ' + images.length);
      var img = document.createElement('img');
      img.src = item.src;
      img.alt = item.alt;
      img.loading = 'lazy';
      img.decoding = 'async';
      button.appendChild(img);
      button.addEventListener('click', function () { open(i); });
      grid.appendChild(button);
    });
  }

  /* Lightbox: a single native <dialog>, created on demand. */
  function ensureLightbox() {
    if (lightbox) return;
    lightbox = document.createElement('dialog');
    lightbox.className = 'photo-lightbox';
    lightbox.setAttribute('aria-label', 'Photo viewer');
    lightbox.innerHTML =
      '<button class="lightbox-close" type="button" aria-label="Close">×</button>' +
      '<button class="lightbox-nav lightbox-prev" type="button" aria-label="Previous photo">‹</button>' +
      '<figure><img alt=""><figcaption></figcaption></figure>' +
      '<button class="lightbox-nav lightbox-next" type="button" aria-label="Next photo">›</button>';
    document.body.appendChild(lightbox);
    lightboxImg = lightbox.querySelector('img');
    lightboxCaption = lightbox.querySelector('figcaption');
    lightbox.querySelector('.lightbox-close').addEventListener('click', function () { lightbox.close(); });
    lightbox.querySelector('.lightbox-prev').addEventListener('click', function () { step(-1); });
    lightbox.querySelector('.lightbox-next').addEventListener('click', function () { step(1); });
    lightbox.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowLeft') { e.preventDefault(); step(-1); }
      if (e.key === 'ArrowRight') { e.preventDefault(); step(1); }
    });
    lightbox.addEventListener('click', function (e) {
      if (e.target === lightbox) lightbox.close();
    });
  }

  function render() {
    var item = images[current];
    lightboxImg.src = item.src;
    lightboxImg.alt = item.alt;
    lightboxCaption.textContent = item.alt;
    lightboxCaption.hidden = !item.alt;
  }

  function open(i) {
    ensureLightbox();
    current = i;
    render();
    lightbox.showModal();
  }

  function step(delta) {
    if (!images.length) return;
    current = (current + delta + images.length) % images.length;
    render();
  }
})();
