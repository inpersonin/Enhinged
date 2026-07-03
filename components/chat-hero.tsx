"use client"

import { AnimatePresence, motion } from "framer-motion"
import { ArrowUpRight, SendHorizontal, Sparkles } from "lucide-react"
import { useEffect, useMemo, useRef, useState } from "react"

type Message = {
  id: number
  role: "user" | "assistant"
  content: string
  streaming?: boolean
}

const metrics = [
  { label: "params", value: "30.04M" },
  { label: "best val loss", value: "1.4363" },
  { label: "training iters", value: "5000" },
  { label: "context window", value: "256" },
]

const suggestions = [
  "Explain attention in Hinglish",
  "Mujhe ek short reply likh do",
  "How was this model trained?",
  "Ek funny line bolo",
]

const apiBase = process.env.NEXT_PUBLIC_API_URL || "https://inpersonin-enhinged.hf.space"

export function ChatHero() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      role: "assistant",
      content: "Main Enhinged hoon. Hindi, English, ya Hinglish me poochho — main backend se reply launga.",
    },
  ])
  const [input, setInput] = useState("")
  const [pending, setPending] = useState(false)
  const [status, setStatus] = useState("ready")
  const nextId = useRef(2)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const element = scrollRef.current
    if (element) {
      element.scrollTo({ top: element.scrollHeight, behavior: "smooth" })
    }
  }, [messages, pending])

  const conversationHistory = useMemo(
    () =>
      messages.map((message) => ({
        role: message.role,
        content: message.content,
      })),
    [messages],
  )

  const sendMessage = async (rawText: string) => {
    const trimmed = rawText.trim()
    if (!trimmed || pending) return

    const userMessage: Message = { id: nextId.current++, role: "user", content: trimmed }
    const assistantId = nextId.current++
    setInput("")
    setPending(true)
    setStatus("thinking")
    setMessages((current) => [
      ...current,
      userMessage,
      { id: assistantId, role: "assistant", content: "", streaming: true },
    ])

    try {
      const response = await fetch(`${apiBase}/generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: trimmed,
          max_new_tokens: 160,
          temperature: 0.82,
          top_k: 40,
          top_p: 0.95,
          repetition_penalty: 1.08,
          do_sample: true,
          conversation_history: [...conversationHistory, { role: "user", content: trimmed }].slice(-8),
        }),
      })

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`)
      }

      const data: { response?: string } = await response.json()
      const text = (data.response || "No response returned.").trim() || "No response returned."

      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId ? { ...message, content: text, streaming: false } : message,
        ),
      )
      setStatus("connected")
    } catch {
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                content: "Backend unavailable right now. Check the HF Space URL or CORS settings.",
                streaming: false,
              }
            : message,
        ),
      )
      setStatus("error")
    } finally {
      setPending(false)
    }
  }

  return (
    <section id="demo" className="relative min-h-screen overflow-hidden px-5 pt-24 md:px-8 lg:px-10">
      <div className="grid-line absolute inset-0 opacity-25" aria-hidden="true" />
      <motion.div
        aria-hidden="true"
        initial={{ opacity: 0, scale: 0.92 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 1.2, ease: [0.22, 1, 0.36, 1] }}
        className="animate-orb absolute left-[15%] top-[14%] h-[32rem] w-[32rem] rounded-full bg-[radial-gradient(circle,rgba(37,99,235,0.4)_0%,rgba(37,99,235,0.12)_32%,transparent_72%)] blur-3xl"
      />
      <div className="relative mx-auto flex min-h-[calc(100vh-6rem)] max-w-7xl flex-col justify-between gap-10 pb-8">
        <div className="grid gap-10 lg:grid-cols-[1.15fr_0.85fr] lg:items-center">
          <div className="space-y-6">
            <motion.p
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.1 }}
              className="font-mono text-[11px] tracking-[0.35em] text-white/55"
            >
              01 — BRUTAL VOID / GLASS INTERFACE
            </motion.p>

            <motion.h1
              initial={{ opacity: 0, y: 28 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.9, delay: 0.15 }}
              className="max-w-3xl text-balance text-5xl font-light tracking-tight text-white md:text-7xl lg:text-8xl"
            >
              ENHINGED
              <br />
              <span className="italic text-white/90">Hinglish chat</span>
              <br />
              without the scroll.
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.25 }}
              className="max-w-xl text-sm leading-6 text-white/70 md:text-base"
            >
              Brutalist void styling, a restrained glass card, and the live HF Space backend. The input sits in the
              hero itself, so the user can chat immediately.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.32 }}
              id="metrics"
              className="grid max-w-2xl gap-3 sm:grid-cols-2 xl:grid-cols-4"
            >
              {metrics.map((metric) => (
                <div key={metric.label} className="glass-shell rounded-[1.25rem] px-4 py-4">
                  <p className="font-mono text-[10px] tracking-[0.3em] text-white/45 uppercase">{metric.label}</p>
                  <p className="mt-2 text-2xl font-medium tracking-tight text-white">{metric.value}</p>
                </div>
              ))}
            </motion.div>
          </div>

          <motion.div
            initial={{ opacity: 0, y: 30, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 1, delay: 0.25, ease: [0.25, 0.46, 0.45, 0.94] }}
            className="glass-shell relative overflow-hidden rounded-[2rem]"
          >
            <div className="border-b border-white/10 px-5 py-4">
              <div className="flex items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <span className="flex size-10 items-center justify-center rounded-2xl bg-[rgba(37,99,235,0.16)] text-[oklch(0.546_0.245_262.881)]">
                    <Sparkles className="size-4" />
                  </span>
                  <div>
                    <p className="font-mono text-[10px] tracking-[0.3em] text-white/45 uppercase">HF SPACE CHAT</p>
                    <p className="text-sm text-white/80">Model is loaded and ready to answer.</p>
                  </div>
                </div>
                <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 font-mono text-[10px] tracking-[0.25em] text-white/65 uppercase">
                  {status}
                </span>
              </div>
            </div>

            <div ref={scrollRef} className="flex h-[28rem] flex-col gap-4 overflow-y-auto px-5 py-5 md:h-[30rem]">
              <AnimatePresence initial={false}>
                {messages.map((message) => (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 14, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    transition={{ duration: 0.32, ease: [0.22, 1, 0.36, 1] }}
                    className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[88%] rounded-[1.4rem] px-4 py-3 text-sm leading-relaxed md:max-w-[78%] ${
                        message.role === "user"
                          ? "rounded-br-md bg-[oklch(0.546_0.245_262.881)] text-white"
                          : "rounded-bl-md border border-white/10 bg-white/6 text-white/90 backdrop-blur-sm"
                      }`}
                    >
                      {message.content || (message.streaming ? <span className="animate-caret">|</span> : null)}
                      {message.streaming ? <span className="ml-1 inline-block h-4 w-[2px] bg-white/70 align-middle" /> : null}
                    </div>
                  </motion.div>
                ))}

                {pending ? (
                  <motion.div
                    key="thinking"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex justify-start"
                  >
                    <div className="flex items-center gap-1.5 rounded-[1.4rem] rounded-bl-md border border-white/10 bg-white/6 px-4 py-3 backdrop-blur-sm">
                      {[0, 1, 2].map((dot) => (
                        <motion.span
                          key={dot}
                          className="size-1.5 rounded-full bg-[oklch(0.546_0.245_262.881)]"
                          animate={{ y: [0, -4, 0], opacity: [0.45, 1, 0.45] }}
                          transition={{ duration: 0.9, repeat: Infinity, delay: dot * 0.14 }}
                        />
                      ))}
                    </div>
                  </motion.div>
                ) : null}
              </AnimatePresence>
            </div>

            <div className="flex gap-2 overflow-x-auto border-t border-white/10 px-5 py-3">
              {suggestions.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  onClick={() => sendMessage(suggestion)}
                  disabled={pending}
                  className="shrink-0 rounded-full border border-white/10 bg-white/5 px-3.5 py-1.5 font-mono text-[10px] tracking-[0.2em] text-white/65 uppercase transition-all hover:border-white/25 hover:bg-white/10 disabled:opacity-45"
                >
                  {suggestion}
                </button>
              ))}
            </div>

            <form
              className="flex items-center gap-3 border-t border-white/10 px-4 py-4"
              onSubmit={(event) => {
                event.preventDefault()
                void sendMessage(input)
              }}
            >
              <input
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="Type a prompt in Hindi, English, or Hinglish..."
                aria-label="Message Enhinged"
                className="flex-1 rounded-2xl border border-white/10 bg-black/45 px-4 py-3 text-sm text-white outline-none transition-colors placeholder:text-white/35 focus:border-[oklch(0.546_0.245_262.881)]"
              />
              <button
                type="submit"
                disabled={!input.trim() || pending}
                className="flex size-12 shrink-0 items-center justify-center rounded-2xl bg-[oklch(0.546_0.245_262.881)] text-white transition-transform hover:scale-105 disabled:scale-100 disabled:opacity-45"
                aria-label="Send message"
              >
                <SendHorizontal className="size-4" />
              </button>
            </form>
          </motion.div>
        </div>

        <div className="flex flex-col gap-4 border-t border-white/10 pt-4 md:flex-row md:items-end md:justify-between">
          <div className="max-w-xl">
            <p className="font-mono text-[10px] tracking-[0.35em] text-white/45 uppercase">02 — DEPLOYMENT</p>
            <p className="mt-2 text-sm text-white/65">
              Frontend on GitHub Pages. Backend on Hugging Face Spaces. Same theme, same tone, same model.
            </p>
          </div>

          <a
            id="deploy"
            href={apiBase}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 self-start rounded-full border border-white/10 bg-white/5 px-4 py-2 font-mono text-[10px] tracking-[0.3em] text-white/75 uppercase transition-all hover:border-white/25 hover:bg-white/10 md:self-auto"
          >
            Open HF Space
            <ArrowUpRight className="size-3.5" />
          </a>
        </div>
      </div>
    </section>
  )
}