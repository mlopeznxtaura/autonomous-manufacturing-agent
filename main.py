"""
autonomous-manufacturing-agent — Entry Point

AI-native production floor: vision inspection, telemetry, autonomous agent.

Usage:
  python main.py --mode inspect --source 0 --line LINE_A
  python main.py --mode agent --line LINE_A
  python main.py --mode simulate --duration 60 --line LINE_A
  python main.py --mode demo
"""
import argparse
import time
import random
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous Manufacturing Agent")
    parser.add_argument("--mode", required=True,
                        choices=["inspect", "agent", "simulate", "demo"])
    parser.add_argument("--source", default="0", help="Camera source (file, RTSP, or index)")
    parser.add_argument("--line", default="LINE_A", help="Production line ID")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--ollama-model", default="mistral")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--kafka", default="localhost:9092")
    parser.add_argument("--duration", type=int, default=60, help="Simulation duration (seconds)")
    parser.add_argument("--metrics-port", type=int, default=9090)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def try_int(s):
    try:
        return int(s)
    except ValueError:
        return s


def mode_inspect(args):
    from perception.inspection import QualityInspector, InspectionVerdict
    from telemetry.kafka_producer import ManufacturingEventProducer
    from observability.metrics import ProductionMetricsCollector

    inspector = QualityInspector(
        model_path=args.model, line_id=args.line,
        save_frames=True, frames_dir="./inspection_frames"
    )
    producer = ManufacturingEventProducer(bootstrap_servers=args.kafka, line_id=args.line)
    metrics = ProductionMetricsCollector(line_id=args.line, port=args.metrics_port)
    metrics.start_server()

    source = try_int(args.source)
    print(f"[Inspect] Starting inspection on {source} | line={args.line}")

    for result in inspector.inspect_stream(source=source, max_frames=args.max_frames):
        producer.emit_inspection(result)
        metrics.record_inspection(result)
        verdict_sym = {"PASS": "✓", "FAIL": "✗", "QUARANTINE": "?"}.get(result.verdict.value, "?")
        print(f"  [{verdict_sym}] {result.part_id} | {result.n_defects} defects | {result.inference_ms:.1f}ms")

    producer.flush()
    producer.close()
    print(f"
Inspection stats: {inspector.stats()}")


def mode_agent(args):
    from agent.operator import ManufacturingOperatorAgent, ProductionContext
    from telemetry.kafka_producer import ManufacturingEventProducer

    producer = ManufacturingEventProducer(bootstrap_servers=args.kafka, line_id=args.line)

    def on_escalation(reason):
        print(f"
[ESCALATION TRIGGERED] {reason[:200]}")
        producer.emit_alert("escalation", "high", reason[:500])

    agent = ManufacturingOperatorAgent(
        line_id=args.line,
        model=args.ollama_model,
        ollama_url=args.ollama_url,
        escalation_callback=on_escalation,
        kafka_producer=producer,
    )

    events = [
        "Defect rate has risen to 4.2% over the last 15 minutes on LINE_A. What should we do?",
        "Machine M3 is showing temperature spike to 92°C. Normal operating range is 60-80°C.",
        "We've had 3 consecutive QUARANTINE results for misalignment defects on camera CAM_02.",
        "Production throughput dropped from 180 to 140 parts/hour in the last 30 minutes.",
    ]

    print(f"[Agent] Autonomous operator active on {args.line}")
    for event in events:
        print(f"
Event: {event}")
        result = agent.analyze(event)
        print(f"Agent: {result['response'][:300]}")
        if result['should_escalate']:
            print(f"  [ESCALATING] {result['escalation_reason'][:100]}")
        time.sleep(1)

    producer.close()


def mode_simulate(args):
    """Simulate production floor telemetry and inspection events."""
    from telemetry.kafka_producer import ManufacturingEventProducer
    from observability.metrics import ProductionMetricsCollector

    producer = ManufacturingEventProducer(bootstrap_servers=args.kafka, line_id=args.line)
    metrics = ProductionMetricsCollector(line_id=args.line, port=args.metrics_port)
    metrics.start_server()

    rng = random.Random(42)
    start = time.time()
    part_counter = 0
    defect_count = 0

    print(f"[Simulate] Running {args.duration}s production simulation on {args.line}")
    while time.time() - start < args.duration:
        part_counter += 1
        is_defect = rng.random() < 0.03  # 3% base defect rate
        verdict = "FAIL" if is_defect else "PASS"
        if is_defect:
            defect_count += 1

        producer.emit_telemetry(
            machine_id="MACHINE_01",
            metrics={
                "temperature_c": rng.gauss(72, 3),
                "vibration_mm_s": rng.gauss(2.0, 0.3),
                "cycle_time_s": rng.gauss(4.2, 0.1),
                "power_kw": rng.gauss(15, 1),
            }
        )

        metrics.update_defect_rate(defect_count / part_counter)
        metrics.update_throughput(int(part_counter / max(time.time() - start, 1) * 3600))
        metrics.update_machine("MACHINE_01", temp=72.0, vibration=2.0, uptime=98.5)

        if part_counter % 50 == 0:
            elapsed = int(time.time() - start)
            rate = defect_count / part_counter * 100
            print(f"  t={elapsed}s | parts={part_counter} | defect_rate={rate:.1f}%")

        time.sleep(0.1)

    producer.flush()
    producer.close()
    print(f"
Simulation complete: {part_counter} parts, {defect_count} defects ({defect_count/part_counter*100:.1f}%)")


def mode_demo(args):
    print("Running manufacturing agent demo...
")
    args.mode = "agent"
    mode_agent(args)


def main():
    args = parse_args()
    print("=" * 60)
    print("  Autonomous Manufacturing Agent")
    print(f"  Mode: {args.mode.upper()} | Line: {args.line}")
    print("=" * 60)

    dispatch = {
        "inspect": mode_inspect,
        "agent": mode_agent,
        "simulate": mode_simulate,
        "demo": mode_demo,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
