// Offline render of one instrument part (note on/off and controller events) through Apple's sampler with a GarageBand/Logic EXS instrument.
// usage: render_part events.json instrument.exs out.wav seconds
import AVFoundation
let a = CommandLine.arguments
let eventsURL = URL(fileURLWithPath: a[1]), exsURL = URL(fileURLWithPath: a[2]), outURL = URL(fileURLWithPath: a[3])
let seconds = Double(a[4])!
struct Ev: Decodable { let t: Double; let k: String; let n: Int; let v: Int }
let events = try JSONDecoder().decode([Ev].self, from: Data(contentsOf: eventsURL)).sorted { $0.t < $1.t }
let sr = 48000.0
let engine = AVAudioEngine(), sampler = AVAudioUnitSampler()
let fmt = AVAudioFormat(standardFormatWithSampleRate: sr, channels: 2)!
engine.attach(sampler)
engine.connect(sampler, to: engine.mainMixerNode, format: fmt)
do { try engine.enableManualRenderingMode(.offline, format: fmt, maximumFrameCount: 4096) } catch { print("manual mode failed:", error); exit(1) }
do { try sampler.loadInstrument(at: exsURL) } catch { print("instrument load failed:", error); exit(2) }
do { try engine.start() } catch { print("engine start failed:", error); exit(1) }
func render() throws {
    let out = try AVAudioFile(forWriting: outURL, settings: engine.manualRenderingFormat.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
    let chunk: AVAudioFrameCount = 32
    let buf = AVAudioPCMBuffer(pcmFormat: engine.manualRenderingFormat, frameCapacity: 4096)!
    let total = AVAudioFramePosition(seconds * sr)
    var i = 0
    while engine.manualRenderingSampleTime < total {
        let now = Double(engine.manualRenderingSampleTime) / sr
        while i < events.count && events[i].t <= now {
            let e = events[i]
            switch e.k {
            case "on":  sampler.startNote(UInt8(e.n), withVelocity: UInt8(e.v), onChannel: 0)
            case "off": sampler.stopNote(UInt8(e.n), onChannel: 0)
            default:    sampler.sendController(UInt8(e.n), withValue: UInt8(e.v), onChannel: 0)
            }
            i += 1
        }
        let st = try engine.renderOffline(chunk, to: buf)
        if st == .success { try out.write(from: buf) } else if st == .error { print("render error"); exit(3) }
    }
    if #available(macOS 15.0, *) { out.close() }
}
try render()
engine.stop(); print("rendered \(seconds)s, \(events.count) events")
