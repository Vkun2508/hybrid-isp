// Plain-language definitions behind every piece of jargon in the interface (<Term k="psnr">).
export interface Entry { name: string; text: string; better?: string }

export const GLOSSARY = {
  psnr: {
    name: "PSNR (peak signal-to-noise ratio)",
    text: "How close a photo is to the clean reference, in decibels. Each +3 dB halves the remaining error.",
    better: "Higher is better",
  },
  ssim: {
    name: "SSIM (structural similarity)",
    text: "How well edges and textures match the clean reference, from 0 to 1.",
    better: "Higher is better",
  },
  de: {
    name: "ΔE (CIEDE2000 colour error)",
    text: "The average colour difference from the reference. Around 1–2 is hard to see; above 5 is obvious.",
    better: "Lower is better",
  },
  lpips: {
    name: "LPIPS (perceptual distance)",
    text: "How different two images look to a network trained to match human judgements of similarity.",
    better: "Lower is better",
  },
  sigma: {
    name: "Noise σ",
    text: "How grainy the RAW is: the spread of the sensor noise on a 0–1 brightness scale. The router measures it on every photo.",
  },
  ev: {
    name: "EV (exposure value)",
    text: "One EV is a doubling of light. −3 EV means the photo got 8× too little light.",
  },
  wb: {
    name: "White balance",
    text: "Removing the colour cast of the light (warm bulbs, cool shade) so white things look white.",
  },
  raw: {
    name: "RAW",
    text: "The sensor's own readings before any processing: one colour per pixel, in a red-green-green-blue mosaic.",
  },
  isp: {
    name: "ISP (image signal processor)",
    text: "The chain of steps every phone runs to turn RAW sensor data into a finished photo.",
  },
  int8: {
    name: "INT8",
    text: "The network's weights stored as 8-bit integers instead of 32-bit floats: 3.5× smaller, and lighter on a phone's CPU.",
  },
  router: {
    name: "Router",
    text: "Measures noise, exposure and white balance on each photo, then decides how much of the network that photo needs.",
  },
  dr: {
    name: "Dynamic range",
    text: "How many stops separate the darkest and brightest parts of the scene. Very wide scenes get local tone mapping.",
  },
  gt: {
    name: "Ground truth",
    text: "A clean reference of the same scene (for the phone samples, the dataset's long-exposure capture). Scores compare against it.",
  },
} satisfies Record<string, Entry>;

export type TermKey = keyof typeof GLOSSARY;
