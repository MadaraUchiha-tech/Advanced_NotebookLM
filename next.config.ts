const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  devIndicators: false,
  serverExternalPackages: ["pdf-parse", "pdfjs-dist", "@langchain/community", "tiktoken"],
};

export default nextConfig;
