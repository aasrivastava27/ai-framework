import { Client } from "@modelcontextprotocol/client";
import { StdioClientTransport } from "@modelcontextprotocol/client/stdio";

const client = new Client({
    name: "ai-framework-test-client",
    version: "1.0.0"
});

const transport = new StdioClientTransport({
    command: "node",
    args: ["dist/index.js"]
});

try {
    await client.connect(transport);

    const result = await client.listTools();

    for (const tool of result.tools) {
        console.log("\n========================================");
        console.log(`TOOL: ${tool.name}`);
        console.log("========================================");

        console.log("\nDescription:");
        console.log(tool.description);

        console.log("\nInput Schema:");
        console.log(
            JSON.stringify(tool.inputSchema, null, 2)
        );
    }
} catch (error) {
    console.error("MCP connection failed:");
    console.error(error);
} finally {
    await client.close();
}