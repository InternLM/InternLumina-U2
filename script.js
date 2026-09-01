const editingExamples = [
  {
    id: 'soft-scarf', sourceAlt: 'Portrait before adding a scarf', resultAlt: 'Portrait with a soft flowing scarf',
    title: 'Add a soft scarf', instruction: 'Add a soft, flowing scarf around her neck.'
  },
  {
    id: 'futuristic-skyline', sourceAlt: 'Black vessel at dusk', resultAlt: 'Vessel with a futuristic city skyline',
    title: 'Add a futuristic skyline', instruction: 'Add a futuristic city skyline in the background.'
  },
  {
    id: 'butterfly-color', sourceAlt: 'Blue butterfly', resultAlt: 'Vibrant red butterfly',
    title: 'Change butterfly color', instruction: 'Change the blue butterfly to a vibrant red butterfly.'
  },
  {
    id: 'street-art-background', sourceAlt: 'Editorial fashion scene on a white background', resultAlt: 'Editorial fashion scene with street-art background',
    title: 'Replace the backdrop', instruction: 'Replace the minimalist white background with a vibrant street-art mural.'
  },
  {
    id: 'white-flowers', sourceAlt: 'Pink flower', resultAlt: 'White flower',
    title: 'Recolor flowers', instruction: 'Change the pink of the flowers to white.'
  },
  {
    id: 'future-city-car', sourceAlt: 'Car interior in a rural setting', resultAlt: 'Car interior overlooking a futuristic city',
    title: 'Change the environment', instruction: 'Change the exterior environment to a futuristic cityscape at sunset.'
  },
  {
    id: 'watercolor-landscape', sourceAlt: 'Landscape painting', resultAlt: 'Landscape rendered as a watercolor',
    title: 'Painting to watercolor', instruction: 'Change the composition of the painting to a watercolor.'
  },
  {
    id: 'red-hair', sourceAlt: 'Girl standing by the sea', resultAlt: 'Girl with red hair standing by the sea',
    title: 'Personalized red hair', instruction: 'Generate a sunset scene of a girl standing by the sea, but her hair is red.'
  },
  {
    id: 'forest-watercolor', sourceAlt: 'Ink forest illustration', resultAlt: 'Watercolor forest illustration',
    title: 'Ink to watercolor', instruction: 'Transform the ink illustration into a vibrant watercolor painting.'
  },
  {
    id: 'sunset-highway', sourceAlt: 'Highway under an overcast sky', resultAlt: 'Highway under a glowing sunset sky',
    title: 'Add a glowing sunset', instruction: 'Transform the overcast sky into a vibrant, glowing sunset.'
  }
];

const generationExamples = [
  { file: '01_风景城市__007__cfg4.0_seed1967757102_ar9x16.png', alt: 'Art Nouveau San Francisco at sunrise', prompt: 'An ornate Art Nouveau and Art Deco illustration of downtown San Francisco at sunrise, with gilded mosaic surfaces, textile-like patterns, and an amber skyline.' },
  { file: '01_风景城市__082__cfg4.0_seed221016654_ar16x9.png', alt: 'Mediterranean coastal town', prompt: 'A romantic impressionist painting of a sunlit Mediterranean coastal town, a flower-lined promenade, moored wooden boats, a palm tree, and a tranquil bay at golden hour.' },
  { file: '02_植物动物__081__cfg4.0_seed2034560693_ar9x16.png', alt: 'Mediterranean villa with roses', prompt: 'A loose impressionistic watercolor of a Mediterranean white villa with terracotta tiles, stone steps, cascading roses, climbing vines, and an airy powder-blue sky.' },
  { file: '03_人物肖像__055__cfg4.0_seed13171928_ar3x4.png', alt: 'Graphite portrait study', prompt: 'A hyperrealistic graphite portrait study of a young Korean woman in three-quarter view, with diffused side light, visible pencil texture, and a quiet gray background.' },
  { file: '05_食品产品__040__cfg4.5_seed637533324_ar4x3.png', alt: 'Vintage Halloween tablescape', prompt: 'A casual film photograph of a vintage Halloween tablescape with paper-mache pumpkins, stacked books, a brass candlestick, and dried leaves in warm window light.' },
  { file: '05_食品产品__077__cfg3.0_seed1361308449_ar16x9.png', alt: 'Rainy bedroom window still life', prompt: 'A soft oil painting of a rain-soaked bedroom window with an open book, steaming tea, and dried flowers on the sill, in muted cream, walnut, olive, and burgundy tones.' },
  { file: '05_食品产品__094__cfg3.0_seed1301974022_ar3x4.png', alt: 'Dark Romantic portrait', prompt: 'A dramatic Dark Romantic oil portrait of a freckled woman with braided red-brown hair, old books, and wilting hydrangeas, lit in deep chiaroscuro.' },
  { file: 'cfg3.0_seed802262695 copy.png', alt: 'Red-haired woman portrait', prompt: 'A girl with red hair.' },
  { file: 'cfg3.5_seed813877396_ar9x16.png', alt: 'Peonies by a sunlit window', prompt: 'An impressionist oil painting of white jasmine and blush peonies in an ornate vase beside a sunlit window, using rich impasto brushwork and warm afternoon light.' },
  { file: 'cfg3.5_seed1656246583_ar16x9.png', alt: 'Mixed media gold-leaf portrait', prompt: 'A mixed-media portrait of a woman’s face assembled from torn paper, green and blue paint, and gold leaf, with a luminous blue eye against an off-white background.' },
  { file: 'cfg4.0_seed329493369.png', alt: 'Mid-century mountain print', prompt: 'A vintage 1960s American landscape screenprint of a Pacific Northwest lake and snow-capped mountains, using muted teal, cream, mustard, orange, and forest green.' },
  { file: 'cfg4.0_seed718659322_ar4x3.png', alt: 'Farmhouse living room at golden hour', prompt: 'A 35mm film photograph of a sophisticated farmhouse living room at golden hour, with walnut furniture, vinyl records, picture windows, and a California desert view.' },
  { file: 'cfg4.0_seed1856507787_ar16x9.png', alt: 'Chiaroscuro citrus still life', prompt: 'A dramatic oil-painted still life of halved citrus fruit on a weathered wooden table, with rich impasto brushwork and strong chiaroscuro side lighting.' },
  { file: 'cfg4.0_seed1911854177.png', alt: 'Porcelain Japanese theme portrait', prompt: 'A photorealistic portrait with Japanese-inspired patterns and traditional colors, rendered in glossy porcelain with fine cracks glowing with gold liquid.' }
];

const imageUnderstandingExamples = [
  { file: 'selected_image_understanding/images/case01_MMMU_Pro_668.jpg', alt: 'Histology image for seizure diagnosis', question: 'For the 15-month-old patient shown, which etiology best explains the seizures?', answer: 'A. Neuronal migration defect' },
  { file: 'selected_image_understanding/images/case04_MMMU_1571.jpg', alt: 'Painting used for artist attribution', question: 'Who created this painting?', answer: 'A. Paul Cézanne' },
  { file: 'selected_image_understanding/images/case06_MMMU_720.jpg', alt: 'Disease-affected weeds in a field', question: 'How should weeds infected by disease be handled?', answer: 'C. Remove all weeds as they will impact yield' },
  { file: 'selected_image_understanding/images/case09_MMMU_Pro_1391.jpg', alt: 'Electronic circuit and input waveform', question: 'Find the current i(t) for the circuit and input shown.', answer: 'F. (Vₘ / 2) {t cos t + sin t} u(t)' },
  { file: 'selected_image_understanding/images/case12_MMMU_7581.jpg', alt: 'Cholesterol values before and after a low-fat diet', question: 'At the 5% significance level, were cholesterol levels significantly lowered after 12 weeks?', answer: 'A. Insufficient evidence to conclude they were lowered' },
  { file: 'selected_image_understanding/images/case20_MMMU_4192.jpg', alt: 'Course and department entity-relationship diagram', question: 'How many departments can a course be offered by in this ER diagram?', answer: 'B. 1' },
  { file: 'selected_image_understanding/images/case24_MMMU_Pro_130.jpg', alt: 'Magnetic tape and pulley mechanism', question: 'Given the acceleration ratio, determine the radius of the larger pulley.', answer: 'F. rₐ = 4.5 in' },
  { file: 'selected_image_understanding/images/case26_MMMU_Pro_1001.jpg', alt: 'Contrast MRI of the head', question: 'What is the diagnosis indicated by this contrasted head MRI?', answer: 'A. Subependymal giant cell astrocytoma' },
  { file: 'selected_image_understanding/images/case31_MMMU_708.jpg', alt: 'Unusual formations on mountain papaya', question: 'What causes these unusual formations on mountain papaya?', answer: 'C. Biotic' },
  { file: 'selected_image_understanding/images/case37_MMMU_Pro_1685.jpg', alt: 'Carbon dioxide phase diagram', question: 'At 6 atm, what phase changes occur as CO₂ is heated from −100°C to −10°C?', answer: 'D. All of them' }
];

const threeDExamples = [
  { file: 'selected_3d_understanding/02_7d22c31b7af049d69b827b4dc23a04b8/blender_render__02_7d22c31b7af049d69b827b4dc23a04b8.png', alt: '3D building render', question: 'What would be the sequence of actions to create a digital 3D model of this building?', answer: 'Gather photos from multiple angles, reconstruct the building with photogrammetry, refine geometry and materials, add lighting, then render and export the model.' },
  { file: 'selected_3d_understanding/13_0d4e574cdf6f48d79b1fe1337fed5097/blender_render__13_0d4e574cdf6f48d79b1fe1337fed5097.png', alt: '3D room render', question: 'Describe the color palette predominantly used in this room.', answer: 'The room predominantly uses shades of beige, white, and light gray.' },
  { file: 'selected_3d_understanding/17_a581dab363ae4457adc362596b0330af/blender_render__17_a581dab363ae4457adc362596b0330af.png', alt: '3D model with patterned surface', question: 'What pattern can be observed on the flat surface of the model?', answer: 'The flat surface displays a checkered pattern with red and black squares.' },
  { file: 'selected_3d_understanding/24_c13f4ab22c7d4ab0814caba9bbd0c830/blender_render__24_c13f4ab22c7d4ab0814caba9bbd0c830.png', alt: '3D winter landscape render', question: 'What season does this landscape typically represent?', answer: 'This landscape typically represents winter.' }
];

const videoExamples = [
  {
    file: 'assets/examples/video/003_S01E2-1-keyframes.jpg',
    alt: 'Ten ordered keyframes from video 003 S01E2-1',
    questions: [{ question: 'Does the video take place in the ancient age, modern age, or future?', answer: 'Modern age.' }]
  },
  {
    file: 'assets/examples/video/008_NWO-2-keyframes.jpg',
    alt: 'Ten ordered keyframes from video 008 NWO-2',
    questions: [
      { question: 'How many different environments appear in the video?', answer: '5.' },
      { question: 'What is the emotional tone of the video?', answer: 'Tense and serious.' }
    ]
  },
  {
    file: 'assets/examples/video/014_LYA-5-keyframes.jpg',
    alt: 'Ten ordered keyframes from video 014 LYA-5',
    questions: [
      { question: 'Does the video take place in the ancient age, modern age, or future?', answer: 'It happens in the future.' },
      { question: 'Are the characters real people? If not, what are they?', answer: 'They are cartoon characters.' },
      { question: 'Are there more than five different characters, excluding people in the background?', answer: 'Yes.' }
    ]
  },
  {
    file: 'assets/examples/video/036_SHQ-5-keyframes.jpg',
    alt: 'Ten ordered keyframes from video 036 SHQ-5',
    questions: [
      { question: 'Are there more than five different scenes in the video?', answer: 'Yes.' },
      { question: 'Are there more than five different characters in the video?', answer: 'Yes.' },
      { question: 'Does the video take place during the day or at night?', answer: 'Daytime.' }
    ]
  },
  {
    file: 'assets/examples/video/058_BKL-1-keyframes.jpg',
    alt: 'Ten ordered keyframes from video 058 BKL-1',
    questions: [
      { question: 'Are there more than five different places in the video?', answer: 'Yes.' },
      { question: 'Where does the video take place: indoors or outdoors?', answer: 'Outdoors.' }
    ]
  }
];

function escapeHTML(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function cleanText(value) {
  return String(value)
    .replace(/\*\*/g, '')
    .replace(/__/g, '')
    .replace(/`/g, '')
    .replace(/<image\s*\d+>/gi, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function qaMarkup(example, kind) {
  return `<article class="qa-example ${kind}">
    <figure><img src="${encodeURI(example.file)}" alt="${escapeHTML(example.alt)}" loading="lazy" /></figure>
    <div class="qa-copy">
      <p class="qa-label">Question</p>
      <p class="qa-question">${escapeHTML(cleanText(example.question))}</p>
      <p class="qa-label">Final answer</p>
      <p class="qa-answer">${escapeHTML(cleanText(example.answer))}</p>
    </div>
  </article>`;
}

const generationGallery = document.getElementById('generation-gallery');
generationGallery.innerHTML = generationExamples.map((example) => `<figure class="generation-example">
  <img src="${encodeURI(`selected_image_generation/${example.file}`)}" alt="${escapeHTML(example.alt)}" loading="lazy" />
  <figcaption><span>Prompt</span>${escapeHTML(cleanText(example.prompt))}</figcaption>
</figure>`).join('');

const editingGallery = document.getElementById('editing-gallery');
editingGallery.innerHTML = editingExamples.map((example) => `<article class="editing-example">
  <div class="editing-pair">
    <figure><span>Source</span><img src="assets/examples/editing/${example.id}-source.png" alt="${example.sourceAlt}" loading="lazy" /></figure>
    <figure><span>Result</span><img src="assets/examples/editing/${example.id}-result.png" alt="${example.resultAlt}" loading="lazy" /></figure>
  </div>
  <div class="editing-caption"><span>Image editing</span><h4>${example.title}</h4><p>${example.instruction}</p></div>
</article>`).join('');

document.getElementById('image-understanding-gallery').innerHTML = imageUnderstandingExamples
  .map((example) => qaMarkup(example, 'image-question-answer'))
  .join('');

document.getElementById('three-d-gallery').innerHTML = threeDExamples
  .map((example) => qaMarkup(example, 'three-d-question-answer'))
  .join('');

document.getElementById('video-gallery').innerHTML = videoExamples.map((example) => `<article class="video-example">
  <div class="video-sheet" aria-label="Horizontally scrollable video keyframe strip"><span class="video-frame-label" aria-hidden="true">▶ VIDEO KEYFRAMES</span><img src="${encodeURI(example.file)}" alt="${escapeHTML(example.alt)}" loading="lazy" /></div>
  <div class="video-copy">${example.questions.map((item) => `<div class="video-qa">
    <p class="qa-label">Question</p><p class="qa-question">${escapeHTML(cleanText(item.question))}</p>
    <p class="qa-label">Final answer</p><p class="qa-answer">${escapeHTML(cleanText(item.answer))}</p>
  </div>`).join('')}</div>
</article>`).join('');
