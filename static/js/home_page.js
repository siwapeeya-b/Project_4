const WORDS = [
  { word: "ป่วย", category: "feeling", note: "ใช้มือข้างขวากางมือเล็กน้อย(ฝ่ามือคว่ำลง) วางมือพาดหรือแตะที่หน้าผาก", media: "/static/media/01.gif" },
  { word: "รัก", category: "feeling", note: "ใช้มือทั้งสองข้างไขว้กันบนหน้าอก (บริเวณหัวใจ) ฝ่ามือแนบอก", media: "/static/media/02.gif" },
  { word: "เสียใจ", category: "feeling", note: "ใช้มือขวากำมือหลวมๆ พร้อมทำมือวนตามเข็มนาฬิกา บริเวณหน้าอกด้านซ้าย", media: "/static/media/03.gif" },
  { word: "กิน", category: "action", note: "ใช้มือข้างขวา นำนิ้วชี้และนิ้วกลางชิดกัน แตะบริเวณริมฝีปากล่างเบาๆ", media: "/static/media/04.gif" },
  { word: "ให้กำลังใจ", category: "action", note: "กำมือทั้งสองมือชูขึ้นข้างลำตัวระดับหน้าอกหรือไหล่", media: "/static/media/05.gif" },
  { word: "ขอโทษ", category: "action", note: "ใช้มือข้างซ้ายหงายฝ่ามือขึ้น มือข้างขวาทับด้านบนพร้อมขยับมือวนเบาๆ", media: "/static/media/06.gif" },
  { word: "ใคร", category: "question", note: "ใช้มือขวาชี้บริเวณหน้า วนทวนเข็มนาฬิกาบริเวณใบหน้า", media: "/static/media/07.gif" },
  { word: "เข้าใจ", category: "action", note: "ใช้มือขวาชูนิ้วขึ้น-ลง แตะบริเวณด้านข้างศีรษะ", media: "/static/media/08.gif" },
  { word: "ที่ไหน", category: "question", note: "ใช้มือข้างขวาชูนิ้วชี้ขึ้น ตั้งมือไว้ระดับหน้าอก พร้อมขยับนิ้วไป-มา", media: "/static/media/09.gif" },
  { word: "ท่าปกติ", category: "action", note: "ตัวตรงพร้อมกับมือทั้งสองแนบชิดลำตัว", media: "/static/media/10.gif" }
];

const grid = document.getElementById("wordGrid");
const resultCount = document.getElementById("resultCount");
const emptyState = document.getElementById("emptyState");
const searchInput = document.getElementById("searchInput");
const tabs = document.querySelectorAll(".tab");

const modalBackdrop = document.getElementById("modalBackdrop");
const modalMedia = document.getElementById("modalMedia");
const modalWord = document.getElementById("modalWord");
const modalCategory = document.getElementById("modalCategory");
const modalNote = document.getElementById("modalNote");
const modalClose = document.getElementById("modalClose");

const CATEGORY_LABEL = {
  greeting: "ทักทาย",
  feeling: "ความรู้สึก",
  question: "คำถาม",
  action: "การกระทำ",
};

let activeCategory = "all";
let query = "";

function placeholderIcon() {
  return `
    <svg viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 22V11.5a2.5 2.5 0 0 1 5 0V19" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/>
      <path d="M17 19v-3.2a2.3 2.3 0 0 1 4.6 0V19" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/>
      <path d="M21.6 19v-2.6a2.2 2.2 0 0 1 4.4 0V19" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/>
      <path d="M26 19v-1.6a2 2 0 0 1 4 0V23c0 5-3.6 9-9 9h-1.6c-3 0-4.9-1-6.7-3.1L8 22.7c-.9-1-.8-2.4.2-3.2.9-.7 2.2-.6 3 .2l2.8 2.7" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>`;
}

function mediaMarkup(item) {
  if (!item.media) return placeholderIcon();
  const isVideo = /\.(mp4|webm|mov)$/i.test(item.media);
  return isVideo
    ? `<video src="${item.media}" autoPlay muted loop playsinline></video>`
    : `<img src="${item.media}" alt="ท่าภาษามือคำว่า ${item.word}">`;
}

function render() {
  const filtered = WORDS.filter((item) => {
    const matchesCategory = activeCategory === "all" || item.category === activeCategory;
    const matchesQuery = item.word.includes(query.trim());
    return matchesCategory && matchesQuery;
  });

  grid.innerHTML = filtered
    .map(
      (item) => `
      <button class="card" data-index="${WORDS.indexOf(item)}">
        <span class="card-media${item.media ? "" : " is-placeholder"}">${mediaMarkup(item)}</span>
        <span class="card-label">${item.word}</span>
      </button>`
    )
    .join("");

  resultCount.textContent = filtered.length ? `พบ ${filtered.length} คำ` : "";
  emptyState.hidden = filtered.length !== 0;

  grid.querySelectorAll(".card").forEach((card) => {
    card.addEventListener("click", () => openModal(WORDS[Number(card.dataset.index)]));
  });
}

function openModal(item) {
  modalCategory.textContent = CATEGORY_LABEL[item.category] || item.category;
  modalWord.textContent = item.word;
  modalNote.textContent = item.note;
  modalMedia.className = `modal-media${item.media ? "" : " is-placeholder"}`;
  modalMedia.innerHTML = mediaMarkup(item);
  modalBackdrop.classList.add("is-open");

  const video = modalMedia.querySelector("video");
  if (video) video.play().catch(() => {});
}

function closeModal() {
  modalBackdrop.classList.remove("is-open");
  modalMedia.innerHTML = "";
}

modalClose.addEventListener("click", closeModal);
modalBackdrop.addEventListener("click", (e) => {
  if (e.target === modalBackdrop) closeModal();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeModal();
});

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => t.classList.remove("is-active"));
    tab.classList.add("is-active");
    activeCategory = tab.dataset.category;
    render();
  });
});

searchInput.addEventListener("input", (e) => {
  query = e.target.value;
  render();
});

render();