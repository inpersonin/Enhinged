"use client"

import { useEffect, useState } from "react"
import { motion } from "framer-motion"

export function CustomCursor() {
  const [position, setPosition] = useState({ x: 0, y: 0 })

  useEffect(() => {
    const handleMove = (event: PointerEvent) => {
      setPosition({ x: event.clientX, y: event.clientY })
    }

    window.addEventListener("pointermove", handleMove)
    return () => window.removeEventListener("pointermove", handleMove)
  }, [])

  return (
    <>
      <motion.div
        aria-hidden="true"
        className="pointer-events-none fixed left-0 top-0 z-[60] hidden size-5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/60 md:block"
        animate={{ x: position.x, y: position.y }}
        transition={{ type: "spring", stiffness: 220, damping: 24, mass: 0.2 }}
      />
      <motion.div
        aria-hidden="true"
        className="pointer-events-none fixed left-0 top-0 z-[60] hidden size-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white md:block"
        animate={{ x: position.x, y: position.y }}
        transition={{ type: "spring", stiffness: 360, damping: 34, mass: 0.12 }}
      />
    </>
  )
}