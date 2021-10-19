#define _POSIX_C_SOURCE 200112L

#include <vulkan/vulkan.h>

#include <GLFW/glfw3.h>

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iso646.h>

#include <sys/stat.h>

#ifdef NDEBUG
#define DEBUG (false)
#else
#define DEBUG (true)
#endif

#define BIT_SET(x, i) ((x) & (1 << (i)))

#define CLEAN_LABEL(v) clean_ ## v

#define STRING(v) #v

#define ARRAY_SIZE(arr) (sizeof((arr)) / sizeof(*(arr)))

void memswap(void* restrict a, void* restrict b, size_t sz) {
    if (a == b) {
        return;
    }
    unsigned char* ca = a;
    unsigned char* cb = b;
    for (size_t i = 0; i < sz; i++) {
        unsigned char t = ca[i];
        ca[i] = cb[i];
        cb[i] = t;
    }
}

unsigned min(unsigned a, unsigned b) {
    return (a < b) ? a : b;
}

unsigned max(unsigned a, unsigned b) {
    return (a > b) ? a : b;
}

unsigned clamp(unsigned x, unsigned a, unsigned b) {
    x = min(x, a);
    x = max(x, b);
    return x;
}

FILE* getLog() {
    return stdout;
}

FILE* getErrorLog() {
    return stderr;
}

typedef enum {
    FR_SUCCESS = 0,
    FR_OPEN_FAILED,
    FR_SIZE_FAILED,
    FR_READ_FAILED,
} FResult;

FResult getFileSize(FILE* file, size_t* size_ptr) {
    struct stat statbuf;
    if (fstat(fileno(file), &statbuf) or
        !S_ISREG(statbuf.st_mode)) {
        return FR_SIZE_FAILED;
    }
    *size_ptr = statbuf.st_size;
    return FR_SUCCESS;
}

FResult loadBinaryFileFP(FILE* fp, char** buffer_ptr, size_t* size_ptr) {
    size_t size = 0;
    FResult res = getFileSize(fp, &size);
    if (res) {
        return res;
    }

    char* buffer = calloc(size, sizeof(char));

    size_t num_read = fread(buffer, 1, size, fp);
    if (num_read != size or ferror(fp)) {
        free(buffer);
        return FR_READ_FAILED;
    }

    *buffer_ptr = buffer;
    *size_ptr = size;

    return FR_SUCCESS;
}

FResult loadBinaryFile(const char* path, char** buffer_ptr, size_t* size_ptr) {
    FILE* fp = fopen(path, "r");
    if (fp) {
        FResult res = loadBinaryFileFP(fp, buffer_ptr, size_ptr);
        fclose(fp);
        return res;
    }
    else {
        return FR_OPEN_FAILED;
    }
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugErrorCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void* p)
{
    (void)type;
    (void)p;
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        fprintf(getErrorLog(), "VULKAN ERROR: %s \n", callback_data->pMessage);
    }
    return VK_FALSE;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugInfoCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void* p)
{
    (void)severity;
    (void)type;
    (void)p;
    fprintf(getErrorLog(), "VULKAN INFO: %s \n", callback_data->pMessage);
    return VK_FALSE;
}

VkResult createDebugMessenger(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pMessenger
) {
    PFN_vkCreateDebugUtilsMessengerEXT f = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, STRING(vkCreateDebugUtilsMessengerEXT));
    if (f) {
        return f(instance, pCreateInfo, pAllocator, pMessenger);
    }
    else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void destroyDebugMessenger(
    VkInstance instance,
    VkDebugUtilsMessengerEXT messenger,
    const VkAllocationCallbacks* pAllocator
) {
    PFN_vkDestroyDebugUtilsMessengerEXT f = (PFN_vkDestroyDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, STRING(vkDestroyDebugUtilsMessengerEXT));
    if (f) {
        f(instance, messenger, pAllocator);
    }
}

typedef struct {
    VkInstance instance;
    VkDebugUtilsMessengerEXT messenger;
} VulkanInstance;

const char** getInstanceLayers(unsigned* num_layers_ptr) {
#if DEBUG
    static const char* required_layers[] = {
        "VK_LAYER_KHRONOS_validation",
    };
    *num_layers_ptr = ARRAY_SIZE(required_layers);
    return required_layers;
#else
    *num_layers_ptr = 0;
    return NULL;
#endif
}

const char** getInstanceExtensions(unsigned* num_extensions_ptr) {
#if DEBUG
    static const char* required_extensions[] = {
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    };
    enum {
        num_required_extensions = ARRAY_SIZE(required_extensions)
    };
#else
    const char** required_extensions = NULL;
    enum {
        num_required_extensions = 0
    };
#endif

    // TODO: using a static buffer here feels like a hack.
    // But the number of windowing system extensions for GLFW
    // is fixed and small (currently 2).
    enum {buffer_elems = 16};
    static const char* extensions[buffer_elems];

    static_assert(
        num_required_extensions <= buffer_elems,
        "Extensions buffer is too small for required extensions"
    );
    memcpy(extensions, required_extensions, sizeof(*required_extensions) * num_required_extensions);

    unsigned num_display_extensions = 0;
    const char** display_extensions = glfwGetRequiredInstanceExtensions(&num_display_extensions);
    unsigned num_extensions = num_display_extensions + num_required_extensions;
    assert(
        num_extensions <= buffer_elems &&
        "Extensions buffer is too small for display extensions"
    );
    memcpy(
        extensions + num_required_extensions, display_extensions,
        num_display_extensions * sizeof(*display_extensions)
    );

    *num_extensions_ptr = num_display_extensions + num_required_extensions;

    return extensions;
}

bool checkInstanceLayerSupported(
    VkLayerProperties* supported, unsigned num_supported,
    const char* layer 
) {
    for (unsigned i = 0; i < num_supported; i++) {
        if (strcmp(supported[i].layerName, layer) == 0) {
            return true;
        }
    }
    return false;
}

bool checkInstanceLayersSupported(
    VkLayerProperties* supported, unsigned num_supported,
    const char** required, unsigned num_required
) {
    bool all = true;
    for (unsigned i = 0; i < num_required; i++) {
        if (!checkInstanceLayerSupported(supported, num_supported, required[i])) {
            fprintf(getErrorLog(), "Layer %s is not supported\n", required[i]);
            all = false;
        }
    }
    return all;
}

bool instanceLayersSupported(const char** required, unsigned num_required) {
    unsigned num_supported;
    vkEnumerateInstanceLayerProperties(&num_supported, NULL);
    VkLayerProperties* supported = calloc(num_supported, sizeof(*supported));
    vkEnumerateInstanceLayerProperties(&num_supported, supported);
    bool all = checkInstanceLayersSupported(supported, num_supported, required, num_required);
    free(supported);
    return all;
}

bool checkInstanceExtensionSupported(
    VkExtensionProperties* supported, unsigned num_supported,
    const char* layer 
) {
    for (unsigned i = 0; i < num_supported; i++) {
        if (strcmp(supported[i].extensionName, layer) == 0) {
            return true;
        }
    }
    return false;
}

bool checkInstanceExtensionsSupported(
    VkExtensionProperties* supported, unsigned num_supported,
    const char** required, unsigned num_required
) {
    bool all = true;
    for (unsigned i = 0; i < num_required; i++) {
        if (!checkInstanceExtensionSupported(supported, num_supported, required[i])) {
            fprintf(getErrorLog(), "Extension %s is not supported\n", required[i]);
            all =  false;
        }
    }
    return all;
}

bool instanceExtensionsSupported(const char** required, unsigned num_required) {
    unsigned num_supported;
    vkEnumerateInstanceExtensionProperties(NULL, &num_supported, NULL);
    VkExtensionProperties* supported = calloc(num_supported, sizeof(*supported));
    vkEnumerateInstanceExtensionProperties(NULL, &num_supported, supported);
    bool all = checkInstanceExtensionsSupported(supported, num_supported, required, num_required);
    free(supported);
    return all;
}

VkResult createVulkanInstance(VulkanInstance* vi) {
    unsigned num_layers;
    const char** layers = getInstanceLayers(&num_layers);
    for (unsigned i = 0; i < num_layers; i++) {
        fprintf(getLog(), "Require layer %s\n", layers[i]);
    }
    if (!instanceLayersSupported(layers, num_layers)) {
        return VK_ERROR_LAYER_NOT_PRESENT;
    }

    unsigned num_extensions;
    const char** extensions = getInstanceExtensions(&num_extensions);
    for (unsigned i = 0; i < num_extensions; i++) {
        fprintf(getLog(), "Require instance extension %s\n", extensions[i]);
    }
    if (!instanceExtensionsSupported(extensions, num_extensions)) {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    VkDebugUtilsMessengerCreateInfoEXT messenger_create_info = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = NULL,
        .flags = 0,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debugErrorCallback,
        .pUserData = NULL,
    };

    VkInstanceCreateInfo instance_create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = DEBUG ? &messenger_create_info : NULL,
        .flags = 0,
        .pApplicationInfo = NULL,
        .enabledLayerCount = num_layers,
        .ppEnabledLayerNames = layers,
        .enabledExtensionCount = num_extensions,
        .ppEnabledExtensionNames = extensions,
    };

    vkCreateInstance(&instance_create_info, NULL, &vi->instance);
#if DEBUG
    createDebugMessenger(vi->instance, &messenger_create_info, NULL, &vi->messenger);
#endif

    return VK_SUCCESS;
}

void destroyVulkanInstance(VulkanInstance* vi) {
#if DEBUG
    destroyDebugMessenger(vi->instance, vi->messenger, NULL);
#endif
    vkDestroyInstance(vi->instance, NULL);
}

typedef struct {
    GLFWwindow* window;
    VkSurfaceKHR surface;
} VulkanWindow;

VulkanWindow createVulkanWindow(VkInstance instance, unsigned width, unsigned height) {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, false);
    VulkanWindow vw;
    vw.window = glfwCreateWindow(width, height, "Vulkan Tutorial", NULL, NULL);
    glfwCreateWindowSurface(instance, vw.window, NULL, &vw.surface);
    return vw;
}

bool shouldClose(GLFWwindow* window) {
    glfwPollEvents();
    return glfwWindowShouldClose(window);
}

void destroyVulkanWindow(VkInstance instance, VulkanWindow* vw) {
    vkDestroySurfaceKHR(instance, vw->surface, NULL);
    glfwDestroyWindow(vw->window);
}

const char** getGPUExtensions(unsigned* num_extensions_ptr) {
    static const char* required_extensions[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };
    *num_extensions_ptr = ARRAY_SIZE(required_extensions);
    return required_extensions;
}

typedef struct{
    unsigned graphics;
    unsigned present;
    bool has_graphics;
    bool has_present;
} QueueFamilies;

bool getGPUGraphicsQueueFamily(
    const VkQueueFamilyProperties* queues,
    unsigned num_queues, unsigned* graphics_ptr
) {
    for (unsigned i = 0; i < num_queues; i++) {
        if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            *graphics_ptr = i;
            return true;
        }
    }
    return false;
}

bool getGPUPresentQueueFamily(
    VkPhysicalDevice gpu, VkSurfaceKHR surface,
    unsigned num_queues, unsigned* present_ptr
) {
    for (unsigned i = 0; i < num_queues; i++) {
        VkBool32 has_present = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(gpu, i, surface, &has_present);
        if (has_present) {
            *present_ptr = i;
            return true;
        }
    }
    return false;
}

QueueFamilies getGPUQueueFamilies(VkPhysicalDevice gpu, VkSurfaceKHR surface) {
    unsigned num_queues = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &num_queues, NULL);
    VkQueueFamilyProperties* queues = calloc(num_queues, sizeof(*queues));
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &num_queues, queues);

    QueueFamilies queue_families;
    queue_families.has_graphics = getGPUGraphicsQueueFamily(queues, num_queues, &queue_families.graphics);
    queue_families.has_present = getGPUPresentQueueFamily(gpu, surface, num_queues, &queue_families.present);

    free(queues);

    return queue_families;
}

typedef struct {
    VkSurfaceCapabilitiesKHR capabilities;
    VkSurfaceFormatKHR* formats;
    VkPresentModeKHR* present_modes;
    unsigned num_formats;
    unsigned num_present_modes;
} SurfaceInfo;

void freeSurfaceInfo(SurfaceInfo* info) {
    free(info->formats);
    free(info->present_modes);
}

SurfaceInfo getGPUSurfaceInfo(
    VkPhysicalDevice gpu, VkSurfaceKHR surface
) {
    SurfaceInfo info;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu, surface, &info.capabilities);

    vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &info.num_formats, NULL);
    info.formats = calloc(info.num_formats, sizeof(*info.formats));
    vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &info.num_formats, info.formats);

    vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, &info.num_present_modes, NULL);
    info.present_modes = calloc(info.num_present_modes, sizeof(*info.present_modes));
    vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, &info.num_present_modes, info.present_modes);

    return info;
}

const char* gpuName(VkPhysicalDevice gpu) {
    static VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(gpu, &props);
    return props.deviceName;
}

bool extensionSupported(
    VkExtensionProperties* supported, unsigned num_supported,
    const char* extension
) {
    for (unsigned i = 0; i < num_supported; i++) {
        if (strcmp(supported[i].extensionName, extension) == 0) {
            return true;
        }
    }
    return false;
}

bool checkGPUExtensionsSupported(
    VkPhysicalDevice gpu,
    VkExtensionProperties* supported, unsigned num_supported,
    const char** required, unsigned num_required
) {
    bool all = true;
    for (unsigned i = 0; i < num_required; i++) {
        if (!extensionSupported(supported, num_supported, required[i])) {
            fprintf(getErrorLog(), "%s does not have support for extension %s\n", gpuName(gpu), required[i]);
            all = false;
        }
    }
    return all;
}

bool gpuExtensionsSupported(
    VkPhysicalDevice gpu,
    const char** required, unsigned num_required
) {
    unsigned num_supported = 0;
    vkEnumerateDeviceExtensionProperties(gpu, NULL, &num_supported, NULL);
    VkExtensionProperties* supported = calloc(num_supported, sizeof(*supported));
    vkEnumerateDeviceExtensionProperties(gpu, NULL, &num_supported, supported);
    bool all = checkGPUExtensionsSupported(gpu, supported, num_supported, required, num_required);
    free(supported);
    return all;
}

bool gpuSurfaceSupported(const SurfaceInfo* surface_info) {
    return surface_info->num_formats and surface_info->num_present_modes;
}

typedef struct {
    VkPhysicalDevice gpu;
    QueueFamilies queue_families;
    SurfaceInfo surface_info;
} VulkanGPU;

void freeVulkanGPU(VulkanGPU* gpu) {
    freeSurfaceInfo(&gpu->surface_info);
}

bool checkGPU(VulkanGPU* gpu, const char** extensions, unsigned num_extensions) {
    bool good = true;
    if (!gpuExtensionsSupported(gpu->gpu, extensions, num_extensions)) {
        good = false;
    }
    if (!gpuSurfaceSupported(&gpu->surface_info)) {
        fprintf(getErrorLog(), "%s does not have support for the required surface operations\n", gpuName(gpu->gpu));
        good = false;
    }
    if (!gpu->queue_families.has_graphics) {
        fprintf(getErrorLog(), "%s does not have a graphics queue\n", gpuName(gpu->gpu));
        good = false;
    }
    if (!gpu->queue_families.has_present) {
        fprintf(getErrorLog(), "%s does not have a present queue\n", gpuName(gpu->gpu));
        good = false;
    }
    return good;
}

VulkanGPU* getInstanceGPUs(
    VkInstance instance, VkSurfaceKHR surface,
    unsigned* num_vgpus_ptr
) {
    unsigned num_gpus = 0;
    vkEnumeratePhysicalDevices(instance, &num_gpus, NULL);
    VkPhysicalDevice* gpus = calloc(num_gpus, sizeof(*gpus));
    vkEnumeratePhysicalDevices(instance, &num_gpus, gpus);

    VulkanGPU* vgpus = calloc(num_gpus, sizeof(*vgpus));

    for (unsigned i = 0; i < num_gpus; i++) {
        vgpus[i] = (VulkanGPU) {
            .gpu = gpus[i],
            .queue_families = getGPUQueueFamilies(gpus[i], surface),
            .surface_info = getGPUSurfaceInfo(gpus[i], surface),
        };
    }

    free(gpus);

    *num_vgpus_ptr = num_gpus;

    return vgpus;
}

unsigned gpuScore(VulkanGPU gpu) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(gpu.gpu, &props);
    return (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? 2 : 1;
}

VulkanGPU* gpuWithHighestScore(VulkanGPU* gpus, unsigned num_gpus) {
    unsigned max_i = 0;
    for (unsigned i = 1; i < num_gpus; i++) {
        if (gpuScore(gpus[max_i]) < gpuScore(gpus[i])) {
            max_i = i;
        }
    }
    return &gpus[max_i];
}

VkResult selectVulkanGPU(VkInstance instance, VkSurfaceKHR surface, VulkanGPU* gpu) {
    unsigned num_gpus = 0;
    VulkanGPU* gpus = getInstanceGPUs(instance, surface, &num_gpus);

    fprintf(getLog(), "Available GPUs:\n");
    for (unsigned i = 0; i < num_gpus; i++) {
        fprintf(getLog(), "%s\n", gpuName(gpus[i].gpu));
    }

    unsigned num_extensions;
    const char** extensions = getGPUExtensions(&num_extensions);
    for (unsigned i = 0; i < num_extensions; i++) {
        fprintf(getLog(), "Require device extension %s\n", extensions[i]);
    }

    unsigned num_good_gpus = 0;
    for (unsigned i = 0; i < num_gpus; i++) {
        if (checkGPU(&gpus[i], extensions, num_extensions)) {
            memswap(&gpus[num_good_gpus++], &gpus[i], sizeof(VulkanGPU));
        }
    }

    VulkanGPU* max_gpu = NULL;
    VkResult res = VK_SUCCESS;
    if (!num_good_gpus) {
        res = VK_ERROR_FEATURE_NOT_PRESENT;
        fprintf(getErrorLog(), "No suitable GPUs found\n");
    }
    else {
        max_gpu = gpuWithHighestScore(gpus, num_good_gpus);
        *gpu = *max_gpu;
        fprintf(getLog(), "Select %s\n", gpuName(gpu->gpu));
    }

    for (unsigned i = 0; i < num_gpus; i++) {
        if (&gpus[i] != max_gpu) {
            freeVulkanGPU(&gpus[i]);
        }
    }
    free(gpus);

    return res;
}

typedef struct {
    VkQueue graphics;
    VkQueue present;
} Queues;

typedef struct {
    VkDevice device;
    Queues queues;
} VulkanDevice;

unsigned* getQueueFamilyIndices(
    const QueueFamilies* queue_families,
    unsigned* num_queue_family_indices_ptr
) {
    assert(queue_families->has_graphics and queue_families->has_present);

    static unsigned queue_family_indices[2];
    queue_family_indices[0] = queue_families->graphics;
    unsigned n = 1;
    if (queue_families->graphics != queue_families->present) {
        queue_family_indices[1] = queue_families->present;
        n++;
    }
    *num_queue_family_indices_ptr = n;

    return queue_family_indices;
}

VulkanDevice createVulkanDevice(VulkanGPU* gpu) {
    unsigned num_queue_family_indices = 0;
    unsigned* queue_family_indices = getQueueFamilyIndices(&gpu->queue_families, &num_queue_family_indices);

    VkDeviceQueueCreateInfo* queue_create_infos = calloc(num_queue_family_indices, sizeof(*queue_create_infos));
    float priority = 1.0f;
    for (unsigned i = 0; i < num_queue_family_indices; i++) {
        queue_create_infos[i] = (VkDeviceQueueCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .queueFamilyIndex = queue_family_indices[i],
            .queueCount = 1,
            .pQueuePriorities = &priority,
        };
    }

    unsigned num_extensions = 0;
    const char** extensions = getGPUExtensions(&num_extensions);

    VkDeviceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .queueCreateInfoCount = num_queue_family_indices,
        .pQueueCreateInfos = queue_create_infos,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = NULL,
        .enabledExtensionCount = num_extensions,
        .ppEnabledExtensionNames = extensions,
        .pEnabledFeatures = NULL,
    };

    VulkanDevice vd;
    vkCreateDevice(gpu->gpu, &create_info, NULL, &vd.device);
    vkGetDeviceQueue(vd.device, gpu->queue_families.graphics, 0, &vd.queues.graphics);
    vkGetDeviceQueue(vd.device, gpu->queue_families.present, 0, &vd.queues.present);

    free(queue_create_infos);

    return vd;
}

void destroyVulkanDevice(VulkanDevice* vd) {
    vkDestroyDevice(vd->device, NULL);
}

VkSurfaceFormatKHR selectSurfaceFormat(const VkSurfaceFormatKHR* formats, unsigned num_formats) {
    for (unsigned i = 0; i < num_formats; i++) {
        if (formats[i].format == VK_FORMAT_B8G8R8A8_SRGB and
            formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return formats[i];
        }
    }
    return formats[0];
}

VkPresentModeKHR selectPresentMode(const VkPresentModeKHR* present_modes, unsigned num_present_modes) {
    for (unsigned i = 0; i < num_present_modes; i++) {
        if (present_modes[i] == VK_PRESENT_MODE_MAILBOX_KHR) {
            return VK_PRESENT_MODE_MAILBOX_KHR;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D selectSwapchainExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR* capabilities) {
    VkExtent2D special_value = {
        .width = UINT32_MAX,
        .height = UINT32_MAX,
    };
    if (capabilities->currentExtent.width == special_value.width and
        capabilities->currentExtent.height == special_value.height) {
        int w = 0;
        int h = 0;
        glfwGetFramebufferSize(window, &w, &h);
        return (VkExtent2D) {
            .width = clamp(
                w, capabilities->minImageExtent.width, capabilities->maxImageExtent.width
            ),
            .height = clamp(
                h, capabilities->minImageExtent.height, capabilities->maxImageExtent.height
            ),
        };
    } else {
        return capabilities->currentExtent;
    }
}

unsigned selectSwapchainImageCount(const VkSurfaceCapabilitiesKHR* capabilities) {
    unsigned min_images = capabilities->minImageCount;
    unsigned max_images = capabilities->maxImageCount;
    unsigned image_count = min_images + 1;
    if (max_images) {
        image_count = min(image_count, max_images);
    }
    return image_count;
}

VkImage* getSwapchainImages(VkDevice device, VkSwapchainKHR swapchain, unsigned* num_images_ptr) {
    unsigned num_images = 0;
    vkGetSwapchainImagesKHR(device, swapchain, &num_images, NULL);
    VkImage* images = calloc(num_images, sizeof(*images));
    vkGetSwapchainImagesKHR(device, swapchain, &num_images, images);

    *num_images_ptr = num_images;

    return images;
}

typedef struct {
    VkSwapchainKHR swapchain;
    VkSurfaceFormatKHR format;
    VkPresentModeKHR present_mode;
    VkExtent2D extent;
    VkImage* images;
    unsigned num_images;
} VulkanSwapchain;

VulkanSwapchain createVulkanSwapchain(
    VkDevice device, const VulkanWindow* vw,
    const VulkanGPU* gpu
) {
    VulkanSwapchain swapchain = {
        .format = selectSurfaceFormat(gpu->surface_info.formats, gpu->surface_info.num_formats),
        .present_mode = selectPresentMode(gpu->surface_info.present_modes, gpu->surface_info.num_present_modes),
        .extent = selectSwapchainExtent(vw->window, &gpu->surface_info.capabilities),
    };
    unsigned image_count = selectSwapchainImageCount(&gpu->surface_info.capabilities);

    bool exclusive =
        gpu->queue_families.graphics == gpu->queue_families.present;
    unsigned queues[] = {
        gpu->queue_families.graphics, gpu->queue_families.present
    };

    VkSwapchainCreateInfoKHR create_info = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .surface = vw->surface,
        .minImageCount = image_count,
        .imageFormat = swapchain.format.format,
        .imageColorSpace = swapchain.format.colorSpace,
        .imageExtent = swapchain.extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode =
            exclusive ? VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT,
        .queueFamilyIndexCount =
            exclusive ? 0 : ARRAY_SIZE(queues),
        .pQueueFamilyIndices = exclusive ? NULL : queues,
        .preTransform = gpu->surface_info.capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = swapchain.present_mode,
        .clipped = VK_TRUE,
        .oldSwapchain = VK_NULL_HANDLE,
    };
    vkCreateSwapchainKHR(device, &create_info, NULL, &swapchain.swapchain);

    swapchain.images = getSwapchainImages(device, swapchain.swapchain, &swapchain.num_images);

    return swapchain;
}

void destroyVulkanSwapchain(VkDevice device, VulkanSwapchain* vs) {
    free(vs->images);
    vkDestroySwapchainKHR(device, vs->swapchain, NULL);
}

VkPipelineLayout createPipelineLayout(VkDevice device) {
    VkPipelineLayoutCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .setLayoutCount = 0,
        .pSetLayouts = NULL,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = NULL,
    };
    VkPipelineLayout layout = VK_NULL_HANDLE;
    vkCreatePipelineLayout(device, &create_info, NULL, &layout);
    return layout;
}

VkRenderPass createRenderPass(VkDevice device, VkFormat format) {
    VkAttachmentDescription attachment_description = {
        .flags = 0,
        .format = format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };

    VkAttachmentReference attachment_reference = {
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    VkSubpassDescription subpass = {
        .flags = 0,
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .inputAttachmentCount = 0,
        .pInputAttachments = NULL,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachment_reference,
        .pResolveAttachments = NULL,
        .pDepthStencilAttachment = NULL,
        .preserveAttachmentCount = 0,
        .pPreserveAttachments = NULL,
    };

    VkSubpassDependency dependency = {
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dependencyFlags = 0,
    };

    VkRenderPassCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .attachmentCount = 1,
        .pAttachments = &attachment_description,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency,
    };

    VkRenderPass render_pass = VK_NULL_HANDLE;
    vkCreateRenderPass(device, &create_info, NULL, &render_pass);

    return render_pass;
}

VkShaderModule createShaderModule(VkDevice device, const char* binary, size_t binary_size) {
    VkShaderModuleCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .codeSize = binary_size,
        .pCode = (const uint32_t*) binary,
    };
    VkShaderModule shader_module;
    vkCreateShaderModule(device, &create_info, NULL, &shader_module);
    return shader_module;
}

VkPipelineShaderStageCreateInfo getPipelineShaderStageCreateInfo(
    VkShaderStageFlagBits stage, VkShaderModule shader_module)
{
    return (VkPipelineShaderStageCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .stage = stage,
        .module = shader_module,
        .pName = "main",
        .pSpecializationInfo = NULL,
    };
}

typedef struct {
    float pos[2];
    float color[3];
} Vertex;

VkVertexInputBindingDescription getVertexBindingDescription() {
    return (VkVertexInputBindingDescription) {
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };
}

const VkVertexInputAttributeDescription* getVertexAttributeDescriptions(unsigned* num_desc_ptr) {
    static VkVertexInputAttributeDescription descs[] = {
        {
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32_SFLOAT,
            .offset = offsetof(Vertex, pos),
        },
        {
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, color),
        },
    };
    *num_desc_ptr = ARRAY_SIZE(descs);
    return descs;
}

VkPipeline createPipeline(
    VkDevice device, VkExtent2D swapchain_extent,
    VkRenderPass render_pass)
{
    const char* vert_name = "shader.vert.spv";
    const char* frag_name = "shader.frag.spv";
    char* vert_binary;
    size_t vert_binary_size;
    if (loadBinaryFile(vert_name, &vert_binary, &vert_binary_size) != FR_SUCCESS) {
        fprintf(getErrorLog(), "Failed to load binary from %s\n", vert_name);
        return VK_NULL_HANDLE;
    }
    char* frag_binary;
    size_t frag_binary_size;
    if (loadBinaryFile(frag_name, &frag_binary, &frag_binary_size) != FR_SUCCESS) {
        fprintf(getErrorLog(), "Failed to load binary from %s\n", vert_name);
        return VK_NULL_HANDLE;
    }

    VkShaderModule vert_module = createShaderModule(device, vert_binary, vert_binary_size);
    VkShaderModule frag_module = createShaderModule(device, frag_binary, frag_binary_size);

    VkPipelineShaderStageCreateInfo shader_stages[] = {
        getPipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, vert_module),
        getPipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, frag_module),
    };

    VkVertexInputBindingDescription vertex_binding_description = getVertexBindingDescription();
    unsigned num_vertex_attribute_descriptions = 0;
    const VkVertexInputAttributeDescription* vertex_attribute_descriptions =
        getVertexAttributeDescriptions(&num_vertex_attribute_descriptions);

    VkPipelineVertexInputStateCreateInfo vertex_input_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &vertex_binding_description,
        .vertexAttributeDescriptionCount = num_vertex_attribute_descriptions,
        .pVertexAttributeDescriptions = vertex_attribute_descriptions,
    };

    VkPipelineInputAssemblyStateCreateInfo input_assembly_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = false,
    };

    VkViewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = swapchain_extent.width,
        .height = swapchain_extent.height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    VkRect2D scissor = {
        .offset = {.x = 0, .y = 0},
        .extent = swapchain_extent,
    };

    VkPipelineViewportStateCreateInfo viewport_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };

    VkPipelineRasterizationStateCreateInfo rasterization_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .depthClampEnable = false,
        .rasterizerDiscardEnable = false,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = false,
        .lineWidth = 1.0f,
    };

    VkPipelineMultisampleStateCreateInfo multisample_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = false,
    };

    VkPipelineColorBlendAttachmentState color_blend_attachment = {
        .blendEnable = false,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                          VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT |
                          VK_COLOR_COMPONENT_A_BIT,
    };

    VkPipelineColorBlendStateCreateInfo color_blend_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .logicOpEnable = false,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment,
    };

    VkPipelineLayout layout = createPipelineLayout(device);

    VkGraphicsPipelineCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .stageCount = ARRAY_SIZE(shader_stages),
        .pStages = shader_stages,
        .pVertexInputState = &vertex_input_state,
        .pInputAssemblyState = &input_assembly_state,
        .pTessellationState = NULL,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterization_state,
        .pMultisampleState = &multisample_state,
        .pDepthStencilState = NULL,
        .pColorBlendState = &color_blend_state,
        .pDynamicState = NULL,
        .layout = layout,
        .renderPass = render_pass,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };

    VkPipeline pipeline = VK_NULL_HANDLE;
    vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &create_info, NULL, &pipeline);

    vkDestroyPipelineLayout(device, layout, NULL);
    vkDestroyShaderModule(device, vert_module, NULL);
    vkDestroyShaderModule(device, frag_module, NULL);

    free(vert_binary);
    free(frag_binary);

    return pipeline;
}

int getGPUMemoryTypeIndex(VkPhysicalDevice gpu, uint32_t supported_types, VkMemoryPropertyFlags required_properties)
{
    VkPhysicalDeviceMemoryProperties gpu_memory_properties;
    vkGetPhysicalDeviceMemoryProperties(gpu, &gpu_memory_properties);
    for (unsigned i = 0; i < gpu_memory_properties.memoryTypeCount; i++) {
        if (BIT_SET(supported_types, i) and
            (required_properties & gpu_memory_properties.memoryTypes[i].propertyFlags) == required_properties) {
            return i;
        }
    }
    return -1;
}

typedef struct {
    VkBuffer buf;
    VkDeviceMemory mem;
} VulkanBuffer;

void destroyVulkanBuffer(VkDevice device, VulkanBuffer* buffer) {
    vkFreeMemory(device, buffer->mem, NULL);
    vkDestroyBuffer(device, buffer->buf, NULL);
}

VkResult createVulkanBuffer(
    VkPhysicalDevice gpu, VkDevice device,
    VkDeviceSize buffer_size, VkBufferUsageFlags buffer_usage,
    VkMemoryPropertyFlags memory_properties,
    VulkanBuffer* buffer
) {
    buffer->buf = VK_NULL_HANDLE;
    buffer->mem = VK_NULL_HANDLE;
    VkResult res = VK_SUCCESS;

    VkBufferCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .size = buffer_size,
        .usage = buffer_usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    res = vkCreateBuffer(device, &create_info, NULL, &buffer->buf);
    if (res != VK_SUCCESS){
        goto fail;
    }

    VkMemoryRequirements memory_requirements;
    vkGetBufferMemoryRequirements(device, buffer->buf, &memory_requirements);

    int memory_type_i = getGPUMemoryTypeIndex(gpu, memory_requirements.memoryTypeBits, memory_properties);
    if (memory_type_i < 0) {
        goto fail;
    }

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = NULL,
        .allocationSize = memory_requirements.size,
        .memoryTypeIndex = memory_type_i,
    };

    res = vkAllocateMemory(device, &alloc_info, NULL, &buffer->mem);
    if (res != VK_SUCCESS) {
        goto fail;
    }

    res = vkBindBufferMemory(device, buffer->buf, buffer->mem, 0);
    if (res != VK_SUCCESS) {
        goto fail;
    }

    return VK_SUCCESS;

fail:
    destroyVulkanBuffer(device, buffer);
    return res;
}

VkResult copyVulkanBuffer(const VulkanDevice* vd, VkCommandPool cmd_pool, const VulkanBuffer* src, const VulkanBuffer* dst, size_t size) {
    VkResult res = VK_SUCCESS;

    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = NULL,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandPool = cmd_pool,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd_buffer = VK_NULL_HANDLE;
    res = vkAllocateCommandBuffers(vd->device, &alloc_info, &cmd_buffer);
    if (res != VK_SUCCESS) {
        goto fail;
    }

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = NULL,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = NULL,
    };
    res = vkBeginCommandBuffer(cmd_buffer, &begin_info);
    if (res != VK_SUCCESS) {
        goto fail;
    }
    
    VkBufferCopy copy_info = {
        .size = size,
        .srcOffset = 0,
        .dstOffset = 0,
    };
    vkCmdCopyBuffer(cmd_buffer, src->buf, dst->buf, 1, &copy_info);

    res = vkEndCommandBuffer(cmd_buffer);
    if (res != VK_SUCCESS) {
        goto fail;
    }

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = NULL,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = NULL,
        .pWaitDstStageMask = NULL,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd_buffer,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = NULL,
    };
    res = vkQueueSubmit(vd->queues.graphics, 1, &submit_info, VK_NULL_HANDLE);
    if (res != VK_SUCCESS) {
        goto fail;
    }
    res = vkQueueWaitIdle(vd->queues.graphics);
    if (res != VK_SUCCESS) {
        goto fail;
    }

    vkFreeCommandBuffers(vd->device, cmd_pool, 1, &cmd_buffer);

    return VK_SUCCESS;

fail:
    vkFreeCommandBuffers(vd->device, cmd_pool, 1, &cmd_buffer);
    return res;
}

const Vertex* getVertexData(unsigned* num_vertices_ptr) {
    static Vertex vertices[] = {
        {
            {0.0f, -0.5f},
            {1.0f, 0.0f, 0.0f},
        },
        {
            {0.5f,  0.5f},
            {0.0f, 1.0f, 0.0f},
        },
        {
            {-0.5f,  0.5f},
            {0.0f, 0.0f, 1.0f},
        },
    };

    *num_vertices_ptr = ARRAY_SIZE(vertices);

    return vertices;
}

VkResult createVertexBuffer(
    VkPhysicalDevice gpu, const VulkanDevice* vd,
    VkCommandPool cmd_pool,
    const void* data, size_t size, size_t elem_size,
    VulkanBuffer* buffer
) {
    VkResult res = VK_SUCCESS;
    VulkanBuffer staging_buffer;
    *buffer = (VulkanBuffer) {
        .buf = VK_NULL_HANDLE,
        .mem = VK_NULL_HANDLE,
    };

    size_t buffer_size = size * elem_size;
    res = createVulkanBuffer(
        gpu, vd->device,
        buffer_size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &staging_buffer
    );
    if (res != VK_SUCCESS) {
        goto fail;
    }

    res = createVulkanBuffer(
        gpu, vd->device,
        buffer_size,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        buffer
    );

    void* map = NULL;
    res = vkMapMemory(vd->device, staging_buffer.mem, 0, buffer_size, 0, &map);
    if (res != VK_SUCCESS) {
        goto fail;
    }
    memcpy(map, data, buffer_size);
    vkUnmapMemory(vd->device, staging_buffer.mem);

    res = copyVulkanBuffer(vd, cmd_pool, &staging_buffer, buffer, buffer_size);
    destroyVulkanBuffer(vd->device, &staging_buffer);
    if (res != VK_SUCCESS) {
        goto fail;
    }

    return VK_SUCCESS;

fail:
    destroyVulkanBuffer(vd->device, &staging_buffer);
    destroyVulkanBuffer(vd->device, buffer);
    return res;
}

VkImageView* createImageViews(VkDevice device, VkFormat format, const VkImage* images, unsigned num_images) {
    VkImageView* views = calloc(num_images, sizeof(*views));
    for (unsigned i = 0; i < num_images; i++) {
        VkImageViewCreateInfo create_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .image = images[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .components = {
                .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = VK_COMPONENT_SWIZZLE_IDENTITY,
            },
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        vkCreateImageView(device, &create_info, NULL, &views[i]);
    }
    return views;
}

void destroyImageViews(VkDevice device, VkImageView* views, unsigned num_views) {
    for (unsigned i = 0; i < num_views; i++) {
        vkDestroyImageView(device, views[i], NULL);
    }
}

VkFramebuffer* createSwapchainFramebuffers(
    VkDevice device, VkRenderPass render_pass,
    const VkImageView* views, unsigned num_views, VkExtent2D swapchain_extent)
{
    VkFramebuffer* framebuffers = calloc(num_views, sizeof(*framebuffers));
    for (unsigned i = 0; i < num_views; i++) {
        VkFramebufferCreateInfo create_info = {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .renderPass = render_pass,
            .attachmentCount = 1,
            .pAttachments = &views[i],
            .width = swapchain_extent.width,
            .height = swapchain_extent.height,
            .layers = 1,
        };
        vkCreateFramebuffer(device, &create_info, NULL, &framebuffers[i]);
    }
    return framebuffers;
}

void destroyFramebuffers(VkDevice device, VkFramebuffer* fbs, unsigned num_fbs) {
    for (unsigned i = 0; i < num_fbs; i++) {
        vkDestroyFramebuffer(device, fbs[i], NULL);
    }
}

VkCommandPool createCommandPool(VkDevice device, unsigned graphics_queue) {
    VkCommandPoolCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .queueFamilyIndex = graphics_queue,
    };
    VkCommandPool command_pool = VK_NULL_HANDLE;
    vkCreateCommandPool(device, &create_info, NULL, &command_pool);
    return command_pool;
}

VkCommandBuffer* allocateCommandBuffers(
    VkDevice device, VkCommandPool command_pool, unsigned num_command_buffers)
{
    VkCommandBufferAllocateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = NULL,
        .commandPool = command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = num_command_buffers,
    };
    VkCommandBuffer* command_buffers = calloc(num_command_buffers, sizeof(*command_buffers));
    vkAllocateCommandBuffers(device, &create_info, command_buffers);
    return command_buffers;
}

void recordCommandBuffers(
    VkCommandBuffer* command_buffers, unsigned num_command_buffers,
    const VkFramebuffer* framebuffers, 
    VkRenderPass render_pass,
    VkPipeline pipeline,
    VkExtent2D swapchain_extent,
    VkBuffer buffer,
    unsigned num_vertices
) {
    for (unsigned i = 0; i < num_command_buffers; i++) {
        VkCommandBufferBeginInfo begin_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = NULL,
            .flags = 0,
            .pInheritanceInfo = NULL,
        };
        vkBeginCommandBuffer(command_buffers[i], &begin_info);

        VkClearValue clear_color = {
            .color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}
        };
        VkRenderPassBeginInfo render_pass_begin_info = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = NULL,
            .renderPass = render_pass,
            .framebuffer = framebuffers[i],
            .renderArea = {
                .offset = {.x = 0, .y = 0},
                .extent = swapchain_extent,
            },
            .clearValueCount = 1,
            .pClearValues = &clear_color,
        };
        vkCmdBeginRenderPass(command_buffers[i], &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        VkBuffer bind_buffers[] = {buffer};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(command_buffers[i], 0, 1, bind_buffers, offsets);
        vkCmdDraw(command_buffers[i], num_vertices, 1, 0, 0);
        vkCmdEndRenderPass(command_buffers[i]);

        vkEndCommandBuffer(command_buffers[i]);
    }
}

VkSemaphore createSemaphore(VkDevice device) {
    VkSemaphoreCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
    };
    VkSemaphore semaphore = VK_NULL_HANDLE;
    vkCreateSemaphore(device, &create_info, NULL, &semaphore);
    return semaphore;
}

VkSemaphore* createSemaphores(VkDevice device, unsigned num_semaphores) {
    VkSemaphore* semaphores = calloc(num_semaphores, sizeof(*semaphores));
    for (unsigned i = 0; i < num_semaphores; i++) {
        semaphores[i] = createSemaphore(device);
    }
    return semaphores;
}

void destroySemaphores(VkDevice device, VkSemaphore* semaphores, unsigned num_semaphores) {
    for (unsigned i = 0; i < num_semaphores; i++) {
        vkDestroySemaphore(device, semaphores[i], NULL);
    }
}

VkFence createFence(VkDevice device) {
    VkFenceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = NULL,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    VkFence fence;
    vkCreateFence(device, &create_info, NULL, &fence);
    return fence;
}

VkFence* createFences(VkDevice device, unsigned num_fences) {
    VkFence* fences = calloc(num_fences, sizeof(*fences));
    for (unsigned i = 0; i < num_fences; i++) {
        fences[i] = createFence(device);
    }
    return fences;
}

void destroyFences(VkDevice device, VkFence* fences, unsigned num_fences) {
    for (unsigned i = 0; i < num_fences; i++) {
        vkDestroyFence(device, fences[i], NULL);
    }
}

void submitDraw(
    VkQueue queue, VkCommandBuffer command_buffer,
    VkSemaphore wait_semaphore, VkSemaphore signal_semaphore,
    VkFence signal_fence)
{
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = NULL,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &wait_semaphore,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signal_semaphore,
    };
    vkQueueSubmit(queue, 1, &submit_info, signal_fence);
}

void submitPresent(VkQueue queue, VkSwapchainKHR swapchain, unsigned image, VkSemaphore wait_semaphore) {
    VkPresentInfoKHR present_info = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = NULL,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &wait_semaphore,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &image,
        .pResults = NULL,
    };
    vkQueuePresentKHR(queue, &present_info);
}

void mainLoop(
    GLFWwindow* window, VulkanDevice* vd,
    VkSwapchainKHR swapchain,
    VkCommandBuffer* command_buffers, unsigned num_command_buffers)
{
    VkSemaphore* acquire_semaphores = createSemaphores(vd->device, num_command_buffers);
    VkSemaphore* render_semaphores = createSemaphores(vd->device, num_command_buffers);
    // Fences are used so that no more frames are being rendered at the same
    // time than there are swapchain images.
    VkFence* frame_fences = createFences(vd->device, num_command_buffers);
    // The swapchain might return an image whose index differs from the current
    // frame's. In this case waiting on the current frame's fence will not be
    // sufficient, as the image might still be accessed by another frame's commands.
    // Store the fence of the last frame that rendered to an image to wait
    // for all accesses to be done.
    VkFence* image_fences = calloc(num_command_buffers, sizeof(*image_fences));
    for (unsigned i = 0; i < num_command_buffers; i++) {
       image_fences[i] = VK_NULL_HANDLE;
    }

    unsigned frame = 0;
    while (!shouldClose(window)) {
        unsigned image = 0;
        vkAcquireNextImageKHR(vd->device, swapchain, UINT64_MAX, acquire_semaphores[frame], VK_NULL_HANDLE, &image);

        if (image_fences[image] != VK_NULL_HANDLE) {
            vkWaitForFences(vd->device, 1, &image_fences[image], true, UINT64_MAX);
        }
        image_fences[image] = frame_fences[frame];
        vkWaitForFences(vd->device, 1, &frame_fences[frame], true, UINT64_MAX);
        vkResetFences(vd->device, 1, &frame_fences[frame]);

        submitDraw(vd->queues.graphics, command_buffers[image], acquire_semaphores[frame], render_semaphores[frame], frame_fences[frame]);
        submitPresent(vd->queues.present, swapchain, image, render_semaphores[frame]);
        glfwSwapBuffers(window);

        frame = (frame + 1) % num_command_buffers;
    }
    vkDeviceWaitIdle(vd->device);

    destroyFences(vd->device, frame_fences, num_command_buffers);
    destroySemaphores(vd->device, render_semaphores, num_command_buffers);
    destroySemaphores(vd->device, acquire_semaphores, num_command_buffers);

    free(image_fences);
    free(frame_fences);
    free(render_semaphores);
    free(acquire_semaphores);
}

void run() {
    VulkanInstance vi;
    if (createVulkanInstance(&vi) != VK_SUCCESS) { return; }
    VulkanWindow vw = createVulkanWindow(vi.instance, 1280, 720);
    VulkanGPU gpu;
    if (selectVulkanGPU(vi.instance, vw.surface, &gpu) != VK_SUCCESS) {
        goto CLEAN_LABEL(gpu);
    }
    VulkanDevice vd = createVulkanDevice(&gpu);
    VulkanSwapchain swapchain = createVulkanSwapchain(vd.device, &vw, &gpu);

    VkRenderPass render_pass = createRenderPass(vd.device, swapchain.format.format);
    VkPipeline pipeline = createPipeline(vd.device, swapchain.extent, render_pass);
    if (pipeline == VK_NULL_HANDLE) {
        goto CLEAN_LABEL(pipeline);
    }

    VkImageView* views = createImageViews(vd.device, swapchain.format.format, swapchain.images, swapchain.num_images);
    VkFramebuffer* framebuffers = createSwapchainFramebuffers(vd.device, render_pass, views, swapchain.num_images, swapchain.extent);

    VkCommandPool command_pool = createCommandPool(vd.device, gpu.queue_families.graphics);
    unsigned num_vertices = 0;
    const Vertex* vertices = getVertexData(&num_vertices);
    VulkanBuffer vertex_buffer;
    if (createVertexBuffer(gpu.gpu, &vd, command_pool, vertices, num_vertices, sizeof(*vertices), &vertex_buffer) != VK_SUCCESS) {
        goto CLEAN_LABEL(vertex_buffer);
    }
    VkCommandBuffer* command_buffers = allocateCommandBuffers(vd.device, command_pool, swapchain.num_images);
    recordCommandBuffers(command_buffers, swapchain.num_images, framebuffers, render_pass, pipeline, swapchain.extent, vertex_buffer.buf, 3);
    
    mainLoop(vw.window, &vd, swapchain.swapchain, command_buffers, swapchain.num_images);

    vkFreeCommandBuffers(vd.device, command_pool, swapchain.num_images, command_buffers);
    free(command_buffers);
    destroyVulkanBuffer(vd.device, &vertex_buffer);
CLEAN_LABEL(vertex_buffer):
    vkDestroyCommandPool(vd.device, command_pool, NULL);

    destroyFramebuffers(vd.device, framebuffers, swapchain.num_images);
    destroyImageViews(vd.device, views, swapchain.num_images);
    free(framebuffers);
    free(views);

    vkDestroyPipeline(vd.device, pipeline, NULL);
CLEAN_LABEL(pipeline):
    vkDestroyRenderPass(vd.device, render_pass, NULL);

    destroyVulkanSwapchain(vd.device, &swapchain);
    destroyVulkanDevice(&vd);
    freeVulkanGPU(&gpu);
CLEAN_LABEL(gpu):
    destroyVulkanWindow(vi.instance, &vw);
    destroyVulkanInstance(&vi);
}

int main() {
    glfwInit();
    run();
    glfwTerminate();
}
