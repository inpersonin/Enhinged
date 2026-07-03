"use client"

import { useEffect, useState } from "react"
import { motion } from "framer-motion"

const links = [
  { label: "demo", href: "#demo" },
  { label: "metrics", href: "#metrics" },
  { label: "deploy", href: "#deploy" },
]

export function Navbar() {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 18)
    onScroll()
    window.addEventListener("scroll", onScroll)
    return () => window.removeEventListener("scroll", onScroll)
  }, [])

  return (
    <motion.header
      initial={{ y: -90 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
      className={`fixed left-0 right-0 top-0 z-50 transition-all duration-500 ${
        scrolled ? "border-b border-white/10 bg-black/55 backdrop-blur-xl" : ""
      }`}
    >
      <nav className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4 md:px-10 md:py-5">
        <a href="#demo" className="group flex items-center gap-3">
          <span className="font-mono text-[11px] tracking-[0.35em] text-white/70">ENHINGED</span>
          <span className="size-2 rounded-full bg-[oklch(0.546_0.245_262.881)] transition-transform duration-300 group-hover:scale-150" />
        </a>

        <div className="hidden items-center gap-8 md:flex">
          {links.map((link, index) => (
            <a
              key={link.label}
              href={link.href}
              className="group relative font-mono text-[11px] tracking-[0.3em] text-white/60 transition-colors hover:text-white"
            >
              <span className="mr-1 text-[oklch(0.546_0.245_262.881)]">0{index + 1}</span>
              {link.label.toUpperCase()}
              <span className="absolute -bottom-1 left-0 h-px w-0 bg-white transition-all duration-300 group-hover:w-full" />
            </a>
          ))}
        </div>

        <div className="flex items-center gap-3">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[oklch(0.546_0.245_262.881)] opacity-70" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-[oklch(0.546_0.245_262.881)]" />
          </span>
          <span className="font-mono text-[10px] tracking-[0.3em] text-white/55">HF SPACE LIVE</span>
        </div>
      </nav>
    </motion.header>
  )
}