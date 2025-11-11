
// docs/config.js
// Update owner/repo/branch to match your repository before publishing.
window.SN_CONFIG = {
  owner: "YOUR_GITHUB_USERNAME_OR_ORG",
  repo: "YOUR_REPO_NAME",
  branch: "main",

  cameras: [
    { id: "borreguiles", name: "Borreguiles", dir: "images/borreguiles", mapPct: [0.58, 0.70] },
    { id: "stadium",     name: "Stadium",     dir: "images/stadium",     mapPct: [0.46, 0.73] },
    { id: "satelite",    name: "Satelite",    dir: "images/satelite",    mapPct: [0.64, 0.58] },
    { id: "veleta",      name: "Veleta",      dir: "images/veleta",      mapPct: [0.70, 0.42] },
  ],

  pageSize: 48,
};
