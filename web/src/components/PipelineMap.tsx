import type { Route } from "../api";

/** The hybrid pipeline as a diagram. With a route, the path this photo took is lit and the other is dimmed. */
export function PipelineMap({ route, wide }: { route?: Route | null; wide?: boolean }) {
  const path = route?.path ?? null;
  const net = path === null ? null : path !== "fast";
  const tags: string[] = [];
  if (route && route.gain > 1.05) tags.push(`×${route.gain.toFixed(1)} exposure gain`);
  if (route && path === "learned_chroma") tags.push("colour-noise cleanup");
  if (route?.local_tone) tags.push("local tone mapping");
  return (
    <div className={"pmap-wrap" + (wide ? " wide" : "")}>
      <ol className="pmap" aria-label="How the hybrid pipeline works">
        <li className="pnode classical">
          <span className="pk">Sensor RAW</span>
          <span className="pd">Black level removed, Bayer mosaic unified to RGGB</span>
        </li>
        <li className="pnode router">
          <span className="pk">Analyser + router</span>
          <span className="pd">Measures noise, exposure and white balance, then picks a path</span>
        </li>
        <li className="pbranch">
          <div className={"pnode learned" + (net === null ? "" : net ? " on" : " off")}>
            <span className="pk">Learned core {net && <span className="took">this photo</span>}</span>
            <span className="pd">1M-parameter U-Net denoises the RAW; its change is capped by the sensor's noise model</span>
          </div>
          <div className={"pnode classical alt" + (net === null ? "" : !net ? " on" : " off")}>
            <span className="pk">Fast path {net === false && <span className="took">this photo</span>}</span>
            <span className="pd">Clean photos skip the network: guided-filter denoise</span>
          </div>
        </li>
        <li className="pnode classical">
          <span className="pk">Classical finish</span>
          <span className="pd">Exposure gain, white balance, demosaic, colour matrix, tone curve, sharpening</span>
          {tags.length > 0 && <span className="ptags">{tags.map((t) => <span key={t} className="pill">{t}</span>)}</span>}
        </li>
        <li className="pnode out">
          <span className="pk">Photo</span>
        </li>
      </ol>
      <div className="pmap-legend" aria-hidden="true">
        <span><i className="lg classical" />Classical maths</span>
        <span><i className="lg learned" />Neural network</span>
        <span><i className="lg router" />Decision</span>
      </div>
    </div>
  );
}
